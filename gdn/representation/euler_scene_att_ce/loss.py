import torch.nn.functional as F
from pointnet2.utils import pointnet2_utils
from ...detector.utils import *


def focal_loss(pp, yy, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
    '''
    Binary focal loss
    '''

    p = pp.view(pp.size(0), -1)
    y = yy.view(yy.size(0), -1)

    pt = (p*y + (1-p)*(1-y)).clamp(1e-4,1-1e-4)         # pt = p if y > 0 else 1-p
    w = focal_alpha*y + (1-focal_alpha)*(1-y)  # w = alpha if y > 0 else 1-alpha
    w = w * (1-pt)**focal_gamma
    loss = torch.sum(-w*pt.log(), 1) / torch.sum(y, 1).clamp(min=1.0) # In original paper, loss need to be normalized by the number of active anchors

    return loss.mean()


def loss_baseline(y_pred, ind, importance, y_true, foreground_w, cls_w, x_w, y_w, z_w, rot_w, **kwargs):

    pi = torch.acos(torch.zeros(1))[0] * 2
    y_dim = y_true.size() # (B, M, n_pitch, n_yaw, 8)

    y_true_importance = (y_true[...,0]>0).any(-1).any(-1).float().contiguous() # (B, M, n_pitch, n_yaw) -> (B, M, n_pitch) -> (B, M)
    y_true_transpose = y_true.view(y_true.size(0), y_true.size(1), -1).transpose(1, 2).contiguous() # (B, M, n_pitch*n_yaw*8) -> (B, ?, M)
    # (B, ?, M) -> (B, ?, k) -> (B, k, ?)
    y_true = pointnet2_utils.gather_operation(y_true_transpose, ind).transpose(1, 2)
    y_true = y_true.view(y_true.size(0), y_true.size(1), *y_dim[2:]).contiguous() # (B, k, n_pitch, n_yaw, 8)

    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, k, n_pitch, n_yaw)
    accum_p           = torch.sigmoid(y_pred[...,0]) # 0~1

    x_loss     = 0
    y_loss     = 0
    z_loss     = 0
    rot_loss   = 0

    foreground_loss = F.binary_cross_entropy(importance, y_true_importance.clamp(1e-4, 1-1e-4))
    cls_loss        = focal_loss(accum_p, accum_gt.float(), **kwargs)

    if accum_gt.any():

        # y_mask = accum_gt.nonzero(as_tuple=True) # (b, ind, ind_pitch, ind_yaw)
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

        x_loss     = torch.abs(x_p - x_gt).mean()
        y_loss     = torch.abs(y_p - y_gt).mean()
        z_loss     = torch.abs(z_p - z_gt).mean()
        dot_prod   = 1.0 - torch.bmm(rot_p, rot_gt.transpose(1, 2))
        rot_loss   = (torch.sum(dot_prod*dot_prod, dim=(1, 2))**0.5).mean()

        return (
            foreground_loss * foreground_w +
            cls_loss * cls_w +
            x_loss * x_w +
            y_loss * y_w +
            z_loss * z_w +
            rot_loss * rot_w
        ), foreground_loss.item(), cls_loss.item(), x_loss.item(), y_loss.item(), z_loss.item(), rot_loss.item()
    else:
        return (
            foreground_loss * foreground_w +
            cls_loss * cls_w +
            x_loss * x_w +
            y_loss * y_w +
            z_loss * z_w +
            rot_loss * rot_w
        ), foreground_loss.item(), cls_loss.item(), x_loss, y_loss, z_loss, rot_loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        weight = -torch.log(torch.FloatTensor([
            config['foreground_w'], config['cls_w'], config['x_w'], config['y_w'], config['z_w'], config['rot_w']
        ]))
        self.log_vars = nn.Parameter(weight)
    def forward(self, outputs, ind, importance, targets):
        ws = torch.exp(-self.log_vars)
        (loss, foreground_loss, cls_loss, x_loss, y_loss, z_loss,
                rot_loss) = loss_baseline(outputs, ind, importance, targets,
                ws[0],
                ws[1],
                ws[2],
                ws[3],
                ws[4],
                ws[5]
               )
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, foreground_loss, cls_loss, x_loss, y_loss, z_loss, rot_loss, ws, torch.exp(uncert).pow(0.5)
