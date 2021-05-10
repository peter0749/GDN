import torch.nn.functional as F
from ...detector.utils import *


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x),
              "sigmoid": lambda x: torch.sigmoid(-x),
              "cross_entropy": lambda x: -torch.log(torch.sigmoid(x).clamp(1e-8, 1.-1e-8)),
              "hinge": lambda x: F.relu(1.-x),
              "double_hinge": lambda x: F.relu(torch.maximum((1.-x)*0.5, -x)),
              "ramp": lambda x: ((1.-x)*0.5).clamp(0, 1)
              }
    return losses[loss_name]


def build_loss(loss_name):
    if loss_name is None or len(loss_name) == 0:
        def ce_loss(pp, yy):
            return F.binary_cross_entropy(torch.sigmoid(pp), yy)
        return ce_loss
    else:
        loss_f = select_loss(loss_name)
        def pu_loss(pp, yy):
            '''
            Simplified + Batched version of PU Loss
            Minimize the non-negative risk:
            R_pu = pi * R_p^+ + max(0, R_u^- - pi * R_p^-)
            ---
            Reference from:
                Ryuichi Kiryo, Gang Niu, Marthinus C. du Plessis, Masashi Sugiyama,
                "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NIPS 2017
                src: https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py
            '''

            prior = 3e-3
            beta = 0.0

            x = pp.view(pp.size(0), -1) # model confidence
            y = yy.view(yy.size(0), -1) # 0: unlabeled, 1: positive

            positive, unlabeled = y, 1.-y
            n_positive, n_unlabeled = torch.sum(positive, 1).clamp(min=1.), torch.sum(unlabeled, 1).clamp(min=1.)

            x_in = x
            y_positive = loss_f(x_in)
            y_unlabeled = loss_f(-x_in)
            positive_risk = prior * torch.sum(positive * y_positive, 1) / n_positive
            negative_risk = torch.sum(unlabeled * y_unlabeled, 1) / n_unlabeled - prior * torch.sum(positive * y_unlabeled, 1) / n_positive
            objective = torch.where(negative_risk < -beta, positive_risk - beta, positive_risk + negative_risk)

            return objective.mean()
        return pu_loss


def loss_baseline(ce_loss_fn, y_pred, y_true, y_teacher, alpha, consist_r, temp, cls_w, x_w, y_w, z_w, rot_w, **kwargs):

    pi = torch.acos(torch.zeros(1))[0] * 2
    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, M, n_pitch, n_yaw, 1)
    accum_p           = y_pred[...,0]
    soft_pred         = torch.sigmoid(y_pred[...,0] / temp) # 0~1
    soft_target       = torch.sigmoid(y_teacher[...,0] / temp) # 0~1

    if accum_gt.any():

        y_pos_mask = tuple(accum_gt.nonzero().transpose(0, 1).contiguous())
        y_neg_mask = tuple((~accum_gt).nonzero().transpose(0, 1).contiguous())

        y_pred_pos_masked = y_pred[y_pos_mask]
        y_true_pos_masked = y_true[y_pos_mask]

        y_pred_neg_masked = y_pred[y_neg_mask]
        y_teacher_neg_masked = y_teacher[y_neg_mask]

        x_p             = y_pred_pos_masked[...,1]
        y_p             = y_pred_pos_masked[...,2]
        z_p             = y_pred_pos_masked[...,3]
        roll_p            = y_pred_pos_masked[...,4:6]
        pitch_residual_p  = y_pred_pos_masked[...,6] # 0~1
        yaw_residual_p    = y_pred_pos_masked[...,7] # 0~1

        x_gt            = y_true_pos_masked[...,1]
        y_gt            = y_true_pos_masked[...,2]
        z_gt            = y_true_pos_masked[...,3]
        roll_gt           = y_true_pos_masked[...,4:6]
        pitch_residual_gt = y_true_pos_masked[...,6] # 0~1
        yaw_residual_gt   = y_true_pos_masked[...,7] # 0~1

        pitch_gt = y_pos_mask[2].float()*(pi/accum_gt.size(2))-pi/2.0 + pitch_residual_gt*(pi/accum_gt.size(2))
        yaw_gt   = y_pos_mask[3].float()*(pi*2.0/accum_gt.size(3))-pi + yaw_residual_gt*(pi*2.0/accum_gt.size(3))

        pitch_p = y_pos_mask[2].float()*(pi/accum_gt.size(2))-pi/2.0 + pitch_residual_p*(pi/accum_gt.size(2))
        yaw_p   = y_pos_mask[3].float()*(pi*2.0/accum_gt.size(3))-pi + yaw_residual_p*(pi*2.0/accum_gt.size(3))

        eps_tensor = torch.full((1,), 1e-6, dtype=roll_gt.dtype, device=roll_gt.device)
        roll_gt_denominator = torch.where(torch.abs(roll_gt[...,1])>1e-6, roll_gt[...,1], eps_tensor)
        roll_p_denominator = torch.where(torch.abs(roll_p[...,1])>1e-6, roll_p[...,1], eps_tensor)

        roll_gt = torch.atan((1-roll_gt[...,0]) / roll_gt_denominator)
        roll_p  = torch.atan((1-roll_p[...,0]) / roll_p_denominator)

        ### Construct Rotation Matrix ###
        rot_gt = rotation_tensor(roll_gt, pitch_gt, yaw_gt) # (B, 3, 3)
        rot_fix_ind = (rot_gt[:,2,2]<0).nonzero() # (M, 1) or empty

        # Fix gripper direction
        if rot_fix_ind.size(0)>0:
            rot_fix_ind = rot_fix_ind[:,0]
            rot_180 = torch.FloatTensor([[[1,  0,  0],
                                          [0, -1,  0],
                                          [0,  0, -1]]]).to(roll_p.device) # (1, 3, 3)
            rot_180 = rot_180.expand(rot_fix_ind.size(0), 3, 3)
            rot_gt[rot_fix_ind] = torch.bmm(rot_180, rot_gt[rot_fix_ind])
        rot_p  = rotation_tensor(roll_p, pitch_p, yaw_p)
        ### End construct rotation matrix ###

        consist_loss = F.l1_loss(y_pred_neg_masked[...,1:], y_teacher_neg_masked[...,1:])
        kd_loss    = F.binary_cross_entropy(soft_pred, soft_target)
        ce_loss    = ce_loss_fn(accum_p, accum_gt.float())
        cls_loss   = alpha * ce_loss + (1.-alpha) * kd_loss
        x_loss     = torch.abs(x_p - x_gt).mean()
        y_loss     = torch.abs(y_p - y_gt).mean()
        z_loss     = torch.abs(z_p - z_gt).mean()
        dot_prod   = torch.bmm(rot_p, rot_gt.transpose(1, 2)) - torch.eye(3, dtype=rot_p.dtype, device=rot_p.device).unsqueeze(0).expand(*rot_p.size())
        rot_loss   = (torch.sum(dot_prod*dot_prod, dim=(1, 2))**0.5).mean()

        return (
            cls_loss * cls_w +
            consist_loss * consist_r +
            x_loss * x_w +
            y_loss * y_w +
            z_loss * z_w +
            rot_loss * rot_w
        ), ce_loss.item(), kd_loss.item(), consist_loss.item(), x_loss.item(), y_loss.item(), z_loss.item(), rot_loss.item()
    else:
        consist_loss = 0
        kd_loss   = 0
        ce_loss   = 0
        x_loss     = 0
        y_loss     = 0
        z_loss     = 0
        rot_loss   = 0

        return torch.zeros(1).requires_grad_(True), ce_loss, kd_loss, consist_loss, x_loss, y_loss, z_loss, rot_loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        weight = -torch.log(torch.FloatTensor([
            config['cls_w'], config['x_w'], config['y_w'], config['z_w'], config['rot_w']
        ]))
        self.log_vars = nn.Parameter(weight)
        self.kd_alpha = config['kd_alpha']
        self.kd_temp  = config['kd_temp']
        self.consist_r  = config['consist_r']
        if 'pu_loss' in self.config and 'pu_loss_type' in self.config:
            self.ce_loss_fn = build_loss(self.config['pu_loss_type'])
        else:
            self.ce_loss_fn = build_loss(None)
    def forward(self, outputs, targets, teacher):
        ws = torch.exp(-self.log_vars)
        (loss, ce_loss, kd_loss, consist_loss, x_loss, y_loss, z_loss,
         rot_loss) = loss_baseline(self.ce_loss_fn, outputs, targets, teacher,
         self.kd_alpha, self.consist_r, self.kd_temp,
         ws[0],ws[1],ws[2],ws[3],ws[4])
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, ce_loss, kd_loss, consist_loss, x_loss, y_loss, z_loss, rot_loss, ws, torch.exp(uncert).pow(0.5)
