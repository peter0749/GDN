from ...detector.utils import *


def focal_loss(pp, yy, focal_alpha=0.25, focal_gamma=2.0, **kwargs):
    '''
    Binary focal loss
    '''

    p = pp.view(pp.size(0), -1)
    y = yy.view(yy.size(0), -1)

    pt = (p*y + (1-p)*(1-y)).clamp(1e-8,1-1e-8)         # pt = p if y > 0 else 1-p
    w = focal_alpha*y + (1-focal_alpha)*(1-y)  # w = alpha if y > 0 else 1-alpha
    w = w * (1-pt)**focal_gamma
    loss = torch.sum(-w*pt.log(), 1) / torch.sum(y, 1).clamp(min=1.0) # In original paper, loss need to be normalized by the number of active anchors

    return loss.mean()


def loss_baseline(y_pred, y_true, cls_w, x_w, y_w, z_w, rot_w, **kwargs):

    pi = torch.acos(torch.zeros(1))[0] * 2
    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, M, n_pitch, n_yaw, 1)
    accum_p           = torch.sigmoid(y_pred[...,0]) # 0~1

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

        zeros = torch.zeros(1, device=roll_gt.device).expand(roll_gt.size(0)) # prevent NaNs

        roll_gt = torch.where(roll_gt[...,1]==0, zeros, torch.atan((1-roll_gt[...,0]) / roll_gt[...,1]))
        roll_p  = torch.where(roll_p[...,1]==0, zeros, torch.atan((1-roll_p[...,0]) / roll_p[...,1]))

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

        #cls_loss     = F.binary_cross_entropy(accum_p, accum_gt.float())
        cls_loss   = focal_loss(accum_p, accum_gt.float(), **kwargs)
        x_loss     = torch.abs(x_p - x_gt).mean()
        y_loss     = torch.abs(y_p - y_gt).mean()
        z_loss     = torch.abs(z_p - z_gt).mean()
        dot_prod   = 1.0 - torch.bmm(rot_p, rot_gt.transpose(1, 2))
        dot_prod   = torch.bmm(rot_p, rot_gt.transpose(1, 2)) - torch.eye(3, dtype=rot_p.dtype, device=rot_p.device).unsqueeze(0).expand(*rot_p.size())
        rot_loss   = (torch.sum(dot_prod*dot_prod, dim=(1, 2))**0.5).mean()

        return (
            cls_loss * cls_w +
            x_loss * x_w +
            y_loss * y_w +
            z_loss * z_w +
            rot_loss * rot_w
        ), cls_loss.item(), x_loss.item(), y_loss.item(), z_loss.item(), rot_loss.item()
    else:
        #cls_loss     = F.binary_cross_entropy(accum_p, accum_gt.float())
        cls_loss   = focal_loss(accum_p, accum_gt.float(), **kwargs)
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


#def loss_baseline(y_pred, y_true, cls_w, x_w, y_w, z_w, roll_w, pitch_w, yaw_w, **kwargs):
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        weight = -torch.log(torch.FloatTensor([
            config['cls_w'], config['x_w'], config['y_w'], config['z_w'], config['rot_w']
        ]))
        self.log_vars = nn.Parameter(weight)
    def forward(self, outputs, targets):
        ws = torch.exp(-self.log_vars)
        (loss, cls_loss, x_loss, y_loss, z_loss,
                rot_loss) = loss_baseline(outputs, targets,
                ws[0],
                ws[1],
                ws[2],
                ws[3],
                ws[4]
               )
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, cls_loss, x_loss, y_loss, z_loss, rot_loss, ws, torch.exp(uncert).pow(0.5)


