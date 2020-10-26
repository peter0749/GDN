from ...detector.utils import *
import torch.nn.functional as F

def mahalanobis_norm(x, cov_inv, mean):
    '''
    x: (B, N, D)
    cov_inv: (B, D, D)
    mean: (B, 1, D)
    '''
    B = x.size(0)
    N = x.size(1)
    D = x.size(2)
    x_c = x - mean # (B, N, D)
    cov_inv_expand = cov_inv.expand(B, N, D, D)
    # (B, N, 1, D) x (B, N, D, D) x (B, N, D, 1) -> (B, N, 1, 1)
    return torch.squeeze(torch.matmul(torch.matmul(x_c.unsqueeze(2), cov_inv_expand), x_c.unsqueeze(3))) # (B, N)

def pinv(A):
    # A: (B, M, N)
    # (A^TA)^-1A^T
    AT = A.transpose(0,2,1)
    ATA = torch.bmm(AT, A) # (B, N, M) x (B, M, N) -> (B, N, N)
    invATA = ATA.inverse()
    return torch.bmm(invATA, AT)

def loss_baseline(y_pred, y_true, k):
    '''
    y_pred: (B, N, 9) -- xyz + 6d
    y_true: (B, M, 9)
    '''

    B = y_true.size(0)
    N = y_pred.size(1)
    M = y_true.size(1)

    y_true_mean = torch.mean(y_true, dim=1, keepdim=True) # (B, 1, 9)
    y_true_centered = y_true - y_true_mean # (B, M, 9) - (B, 1, 9) -> (B, M, 9)
    cov = torch.bmm(y_true_centered.permute(0, 2, 1), y_true_centered) # (B, 9, M) x (B, M, 9) -> (B, 9, 9)
    cov_inv = pinv(cov) # (B, 9, 9)
    y_pred_expand = y_pred.unsqueeze(1).expand(B, M, N, 9) # (B, 1, N, 9) -> (B, M, N, 9)
    y_true_expand = y_true.unsqueeze(2).expand(B, M, N, 9) # (B, M, 1, 9) -> (B, M, N, 9)
    D_pairwise = y_true_expand - y_pred_expand # (B, M, N, 9)
    D_normalized = mahalanobis_norm(D_pairwise.view(B, M*N, 9), cov_inv, y_true_mean).view(B, M, N) # (B, M*N) -> (B, M, N)
    # Select top-K nearest neighbors between prediction and ground truth for optimization
    D_topK = torch.topk(D_normalized, k, largest=True, sorted=False, dim=1)
    return D_topK.values.mean()


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
    def forward(self, outputs, targets):
        return loss_baseline(outputs, targets, config['loss_topk'])
