from ...detector.utils import *
import torch.nn.functional as F


def pu_loss(prior, loss_f, beta, pp, yy, epsilon=1e-10, **kwargs):
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

    return objective


def loss_baseline(prior, loss_f, y_pred, y_true, cls_w, rot_w, trans_w, **kwargs):

    beta = 0.0

    accum_gt          = (y_true[...,0]>0) # binary: shape: (B, M)
    accum_p           = y_pred[...,0]

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

        cls_loss       = pu_loss(prior, loss_f, beta, accum_p, y_true[...,0].clamp(0,1), **kwargs).mean()
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
        cls_loss = pu_loss(prior, loss_f, beta, accum_p, y_true[...,0].clamp(0,1), **kwargs).mean()
        rot_loss = 0
        trans_loss = 0

        return (
            cls_loss * cls_w +
            rot_loss * rot_w +
            trans_loss * trans_w
        ), cls_loss.item(), rot_loss, trans_loss


def select_loss(loss_name):
    losses = {"logistic": lambda x: F.softplus(-x),
              "sigmoid": lambda x: torch.sigmoid(-x),
              "cross_entropy": lambda x: -torch.log(torch.sigmoid(x).clamp(1e-8, 1.-1e-8)),
              "hinge": lambda x: F.relu(1.-x),
              "double_hinge": lambda x: F.relu(torch.maximum((1.-x)*0.5, -x))
              }
    return losses[loss_name]


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
        self.prior = 0.2 # TODO: Use running means instead
        weight = -torch.log(torch.FloatTensor([
            config['cls_w'], config['rot_w'], config['trans_w']
        ]))
        self.log_vars = nn.Parameter(weight)
        self.loss_f = select_loss(config['pu_loss_type'])
    def forward(self, outputs, targets):
        ws = torch.exp(-self.log_vars)
        (loss, cls_loss, rot_loss, trans_loss) = loss_baseline(self.prior, self.loss_f, outputs, targets, ws[0], ws[1], ws[2])
        uncert = 0.5 * torch.sum(self.log_vars) # regularization
        loss += uncert
        return loss, cls_loss, rot_loss, trans_loss, ws, torch.exp(uncert).pow(0.5)
