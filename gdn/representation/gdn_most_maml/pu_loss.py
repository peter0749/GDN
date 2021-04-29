from ...detector.utils import *
import torch.nn.functional as F


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x),
              "sigmoid": lambda x: torch.sigmoid(-x),
              "cross_entropy": lambda x: -torch.log(torch.sigmoid(x).clamp(1e-8, 1.-1e-8)),
              "hinge": lambda x: F.relu(1.-x),
              "double_hinge": lambda x: F.relu(torch.maximum((1.-x)*0.5, -x))
              }
    return losses[loss_name]


def pu_loss(prior, loss_f, beta, pp, yy, **kwargs):
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


def loss_baseline(prior, loss_f, y_pred, y_true, cls_w, x_w, y_w, z_w, rot_w, **kwargs):

    beta = 0.0
    pi = torch.acos(torch.zeros(1))[0] * 2
    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, M, n_pitch, n_yaw, 1)
    accum_p           = y_pred[...,0]

    if accum_gt.any():

        y_mask = tuple(accum_gt.nonzero().transpose(0, 1).contiguous())
        y_pred_masked = y_pred[y_mask]
        y_true_masked = y_true[y_mask]

        x_p             = y_pred_masked[...,1]
        y_p             = y_pred_masked[...,2]
        z_p             = y_pred_masked[...,3]
        roll_p            = y_pred_masked[...,4:6]
        pitch_residual_p  = y_pred_masked[...,6] # 0~1
        yaw_residual_p    = y_pred_masked[...,7] # 0~1

        x_gt            = y_true_masked[...,1]
        y_gt            = y_true_masked[...,2]
        z_gt            = y_true_masked[...,3]
        roll_gt           = y_true_masked[...,4:6]
        pitch_residual_gt = y_true_masked[...,6] # 0~1
        yaw_residual_gt   = y_true_masked[...,7] # 0~1

        pitch_gt = y_mask[2].float()*(pi/accum_gt.size(2))-pi/2.0 + pitch_residual_gt*(pi/accum_gt.size(2))
        yaw_gt   = y_mask[3].float()*(pi*2.0/accum_gt.size(3))-pi + yaw_residual_gt*(pi*2.0/accum_gt.size(3))

        pitch_p = y_mask[2].float()*(pi/accum_gt.size(2))-pi/2.0 + pitch_residual_p*(pi/accum_gt.size(2))
        yaw_p   = y_mask[3].float()*(pi*2.0/accum_gt.size(3))-pi + yaw_residual_p*(pi*2.0/accum_gt.size(3))

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

        cls_loss   = pu_loss(prior, loss_f, beta, accum_p, accum_gt.float(), **kwargs)
        x_loss     = torch.abs(x_p - x_gt).mean()
        y_loss     = torch.abs(y_p - y_gt).mean()
        z_loss     = torch.abs(z_p - z_gt).mean()
        dot_prod   = 1.0 - torch.bmm(rot_p, rot_gt.transpose(1, 2))
        rot_loss   = (torch.sum(dot_prod*dot_prod, dim=(1, 2))**0.5).mean()

        return (
            cls_loss * cls_w +
            x_loss * x_w +
            y_loss * y_w +
            z_loss * z_w +
            rot_loss * rot_w
        ), cls_loss.item(), x_loss.item(), y_loss.item(), z_loss.item(), rot_loss.item()
    else:
        cls_loss   = pu_loss(prior, loss_f, beta, accum_p, accum_gt.float(), **kwargs)
        x_loss     = 0
        y_loss     = 0
        z_loss     = 0
        rot_loss   = 0

        return (
            cls_loss * cls_w +
            x_loss * x_w +
            y_loss * y_w +
            z_loss * z_w +
            rot_loss * rot_w
        ), cls_loss.item(), x_loss, y_loss, z_loss, rot_loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        weight = -torch.log(torch.FloatTensor([
            config['cls_w'], config['x_w'], config['y_w'], config['z_w'], config['rot_w']
        ]))
        self.log_vars = nn.Parameter(weight)
        self.prior = 3e-3 # TODO: Use running means instead
        self.loss_f = select_loss(config['pu_loss_type'])
    def forward(self, outputs, targets):
        ws = torch.exp(-self.log_vars)
        (loss, cls_loss, x_loss, y_loss, z_loss,
                rot_loss) = loss_baseline(self.prior, self.loss_f, outputs, targets,
                ws[0],
                ws[1],
                ws[2],
                ws[3],
                ws[4]
               )
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, cls_loss, x_loss, y_loss, z_loss, rot_loss, ws, torch.exp(uncert).pow(0.5)
