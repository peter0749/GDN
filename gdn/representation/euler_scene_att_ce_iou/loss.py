import torch.nn.functional as F
from pointnet2.utils import pointnet2_utils
from ...detector.utils import *
from torch.autograd import Variable

def affine_inv(f):
    # f: (B, 3, 4)
    R_inv = f[:,:3,:3].transpose(1, 2) # (B, 3, 3)
    T_inv = -torch.bmm(R_inv, f[:,:3,3:4]) # (B, 3, 3) x (B, 3, 1) -> (B, 3, 1)
    f_inv = torch.cat((R_inv, T_inv), 2)
    h = torch.tensor([0, 0, 0, 1], requires_grad=True, dtype=f.dtype, device=f.device).view(1, 1, 4).expand(f.size(0), 1, 4)
    f_inv = torch.cat((f_inv, h), 1)
    return f_inv

def predIoU(net, pred, target, iou_rescale=1.0):
    pred_iou = net(pred, target)[-1] # (B, 1)
    return pred_iou.clamp(0, 1) ** (1. / iou_rescale)

class IoUNet(nn.Module):
    def __init__(self, n_bins):
        super(IoUNet, self).__init__()
        self.n_bins = n_bins
        self.piecewise_linear_base = torch.linspace(0, 1, steps=n_bins+1, dtype=torch.float)[:-1] # (n_bins,)
        self.register_buffer('linspace_constant', self.piecewise_linear_base)
        self.backbone = nn.Sequential(
            nn.Linear(9, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.cls_branch = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.n_bins),
            nn.LogSoftmax(dim=1)
        )
        self.reg_branch = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.n_bins)
        )

    def forward(self, pred, target):
        # Preprocess
        A = torch.bmm(pred, affine_inv(target)) # Affine transform: target -> pred
        d6 = A[:,:,:2].reshape(A.size(0), 6) # (B, 6)
        t = A[:,:3, 3].view(A.size(0), 3) # (B, 3)
        x = torch.cat((d6, t), 1).float().cuda() # (B, 9)

        # Learning
        feature = self.backbone(x)
        match_bin = self.cls_branch(feature)  # (B, n_bins)
        iou_scores = self.reg_branch(feature) # (B, n_bins)
        iou_scores = iou_scores + Variable(self.linspace_constant).unsqueeze(0).expand(iou_scores.size(0), self.n_bins)
        select = torch.max(match_bin, 1, keepdim=True)[1]
        iou_selected = torch.gather(iou_scores, 1, select)
        return match_bin, iou_scores, iou_selected

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


def loss_baseline(config, iounet, point_cloud, y_pred, ind, importance, y_true, foreground_w, cls_w, iou_w, **kwargs):

    pi = torch.acos(torch.zeros(1))[0] * 2
    y_dim = y_true.size() # (B, M, n_pitch, n_yaw, 8)

    y_true_importance = (y_true[...,0]>0).any(-1).any(-1).float().contiguous() # (B, M, n_pitch, n_yaw) -> (B, M, n_pitch) -> (B, M)
    y_true_transpose = y_true.view(y_true.size(0), y_true.size(1), -1).transpose(1, 2).contiguous() # (B, M, n_pitch*n_yaw*8) -> (B, ?, M)
    point_cloud_t = point_cloud.transpose(1, 2).contiguous() # (B, 3, N)
    # (B, ?, M) -> (B, ?, k) -> (B, k, ?)
    y_true = pointnet2_utils.gather_operation(y_true_transpose, ind).transpose(1, 2)
    keypoints = pointnet2_utils.gather_operation(point_cloud_t, ind).transpose(1, 2) # (B, k, 3)
    y_true = y_true.view(y_true.size(0), y_true.size(1), *y_dim[2:]).contiguous() # (B, k, n_pitch, n_yaw, 8)

    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, k, n_pitch, n_yaw)
    accum_p           = torch.sigmoid(y_pred[...,0]) # 0~1

    foreground_loss = F.binary_cross_entropy(importance, y_true_importance)
    cls_loss        = focal_loss(accum_p, accum_gt.float(), **kwargs)

    if accum_gt.any():

        # y_mask = accum_gt.nonzero(as_tuple=True) # (b, ind, ind_pitch, ind_yaw)
        y_mask = tuple(accum_gt.nonzero().transpose(0, 1).contiguous())
        y_pred_masked = y_pred[y_mask]
        y_true_masked = y_true[y_mask]
        keypoints_us = keypoints.unsqueeze(2).unsqueeze(2) # (B, k, 1, 1, 3)
        # (B, k, n_pitch, n_yaw, 3) -> (?, 3)
        keypoints_masked = keypoints_us.expand(y_true.size(0), y_true.size(1), y_true.size(2), y_true.size(3), 3)[y_mask]

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

        ## Construct Translation Matrix ##
        x_p_gripper_frame = x_p.unsqueeze(-1) * config['hand_height']
        y_p_gripper_frame = (y_p-0.5).unsqueeze(-1) * config['gripper_width']
        z_p_gripper_frame = (z_p-0.5).unsqueeze(-1) * config['thickness_side'] # (?, 1)
        T_p_gripper_frame = torch.cat((x_p_gripper_frame, y_p_gripper_frame, z_p_gripper_frame), -1) # (?, 3)
        T_p_world = keypoints_masked.unsqueeze(-1) - torch.bmm(rot_p, T_p_gripper_frame.unsqueeze(-1)) # (?, 3, 1)

        x_gt_gripper_frame = x_gt.unsqueeze(-1) * config['hand_height']
        y_gt_gripper_frame = (y_gt-0.5).unsqueeze(-1) * config['gripper_width']
        z_gt_gripper_frame = (z_gt-0.5).unsqueeze(-1) * config['thickness_side']
        T_gt_gripper_frame = torch.cat((x_gt_gripper_frame, y_gt_gripper_frame, z_gt_gripper_frame), -1)
        T_gt_world = keypoints_masked.unsqueeze(-1) - torch.bmm(rot_gt, T_gt_gripper_frame.unsqueeze(-1))
        ## End Construct Translation Matrix ##

        ## Transformation Matrix ##
        P_pred = torch.cat((rot_p, T_p_world), 2) # (?, 3, 4)
        P_gt = torch.cat((rot_gt, T_gt_world), 2) # (?, 3, 4)
        iou_loss = (1.0 - iounet(P_pred, P_gt)[-1].clamp(max=1.0)).mean()
        ## Transformation Matrix ##

        return (
            foreground_loss * foreground_w +
            cls_loss * cls_w +
            iou_loss * iou_w
        ), foreground_loss.item(), cls_loss.item(), iou_loss.item()
    else:
        return (
            foreground_loss * foreground_w +
            cls_loss * cls_w
        ), foreground_loss.item(), cls_loss.item(), 0


#def loss_baseline(config, iounet, point_cloud, y_pred, ind, importance, y_true, foreground_w, cls_w, iou_w, **kwargs):
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        self.iounet = IoUNet(config['iounet_nbins']).cuda()
        self.iounet.load_state_dict(torch.load(config['iounet_weights']))
        self.iounet = self.iounet.eval()
        for iounet_param in self.iounet.parameters():
            iounet_param.requires_grad = False
        weight = -torch.log(torch.FloatTensor([
            config['foreground_w'], config['cls_w'], config['iou_w']
        ]))
        self.log_vars = nn.Parameter(weight)
    def forward(self, point_cloud, outputs, ind, importance, targets):
        ws = torch.exp(-self.log_vars)
        (loss, foreground_loss, cls_loss, iou_loss) = loss_baseline(
            self.config,
            self.iounet,
            point_cloud,
            outputs,
            ind,
            importance,
            targets,
            ws[0],ws[1],ws[2]
        )
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, foreground_loss, cls_loss, iou_loss, ws, torch.exp(uncert).pow(0.5)
