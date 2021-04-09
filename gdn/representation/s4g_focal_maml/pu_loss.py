from ...detector.utils import *
import torch.nn.functional as F


def pu_loss(prior, pp, yy, epsilon=1e-10, **kwargs):
    '''
    Reference from:
        Yuewei Yang and Kevin J Liang, "Object Detection as a Positive-Unlabeled Problem", NIPS 2020
        http://people.ee.duke.edu/~lcarin/0264.pdf
    '''

    c = pp.view(pp.size(0), -1) # model confidence
    y = yy.view(yy.size(0), -1) # 0: unlabeled, 1: positive

    # Number of positive, unlabeled
    Np, Nu = torch.sum(y, 1).clamp(min=1.), torch.sum(1.-y, 1).clamp(min=1.)

    # minimize following risks
    log_c_p = torch.log(c+epsilon)
    log_c_u = torch.log(1.-c+epsilon)
    Rp_plus = -torch.sum(y*log_c_p, 1) / Np
    Rp_neg  = -torch.sum((1.-y)*log_c_p, 1) / Np
    Ru_neg  = -torch.sum((1.-y)*log_c_u, 1) / Nu

    # non-negative PU risk estimator (constraint negative risk >= 0)
    pu_risk = prior * Rp_plus + (Ru_neg - prior * Rp_neg).clamp(min=0.)

    return pu_risk.mean()


def loss_baseline(prior, y_pred, y_true, cls_w, rot_w, trans_w, **kwargs):

    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, M)
    accum_p           = torch.sigmoid(y_pred[...,0]) # 0~1

    if accum_gt.any():

        y_mask = tuple(accum_gt.nonzero().transpose(0, 1).contiguous())
        y_pred_masked = y_pred[y_mask]
        y_true_masked = y_true[y_mask]

        rot_p     = y_pred_masked[...,1:10].view(y_pred_masked.size(0), 3, 3)
        rot_gt    = y_true_masked[...,1:10].view(y_true_masked.size(0), 3, 3) # (N, 3, 3)

        trans_p   = y_pred_masked[...,10:13] # (N, 3)
        trans_gt  = y_true_masked[...,10:13] # (N, 3)

        rot_180 = torch.FloatTensor([[[1,  0,  0],
                                      [0, -1,  0],
                                      [0,  0, -1]]]).to(rot_p.device) # (1, 3, 3)
        rot_180 = rot_180.expand(rot_gt.size(0), 3, 3) # (N, 3, 3)

        cls_loss       = pu_loss(prior, accum_p, accum_gt.float(), **kwargs)
        rot_loss_fold1 = (rot_p - rot_gt).pow(2).view(rot_gt.size(0), -1).mean(1) # (N, 3, 3) -> (N, 9) -> (N,)
        rot_loss_fold2 = (rot_p - torch.bmm(rot_180, rot_gt)).pow(2).view(rot_gt.size(0), -1).mean(1)
        rot_loss = torch.min(rot_loss_fold1, rot_loss_fold2).mean()
        trans_loss = (trans_p-trans_gt).pow(2).mean(1).mean()

        return (
            cls_loss * cls_w +
            rot_loss * rot_w +
            trans_loss * trans_w
        ), cls_loss.item(), rot_loss.item(), trans_loss.item()
    else:
        cls_loss = pu_loss(prior, accum_p, accum_gt.float(), **kwargs)
        rot_loss = 0
        trans_loss = 0

        return (
            cls_loss * cls_w +
            rot_loss * rot_w +
            trans_loss * trans_w
        ), cls_loss.item(), rot_loss, trans_loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        self.prior = 0.1 # TODO: Use running means instead
        weight = -torch.log(torch.FloatTensor([
            config['cls_w'], config['rot_w'], config['trans_w']
        ]))
        self.log_vars = nn.Parameter(weight)
    def forward(self, outputs, targets):
        ws = torch.exp(-self.log_vars)
        (loss, cls_loss, rot_loss, trans_loss) = loss_baseline(self.prior, outputs, targets, ws[0], ws[1], ws[2])
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, cls_loss, rot_loss, trans_loss, ws, torch.exp(uncert).pow(0.5)
