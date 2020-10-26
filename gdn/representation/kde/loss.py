from ...detector.utils import *
import torch.nn.functional as F

def mahalanobis_norm(x, cov_inv, mean):
    '''
    x: (N', N, D)
    cov_inv: (D, D)
    mean: (1, D)
    '''
    x_c = x - mean.unsqueeze(0) # (N', N, D)
    cov_inv_expand = cov_inv.expand(x_c.size(0), x_c.size(1), cov_inv.size(0), cov_inv.size(1)) # (N', N, D, D)
    # (N', N, 1, D) x (N', N, D, D) x (N', N, D, 1) -> (N', N, 1, 1)
    return torch.squeeze(torch.matmul(torch.matmul(x_c.unsqueeze(2), cov_inv_expand), x_c.unsqueeze(3))) # (N', N)

def loss_baseline(y_pred, y_true, mask, k, max_pair):
    '''
    y_pred: (B, N, 9) -- xyz + 6d
    y_true: (B, M, 9)
    mask: (B, M) to compute gradient
    '''

    y_pred = y_pred[:,:max_pair] # Prevent OOM

    B = y_true.size(0)
    N = y_pred.size(1)
    M = y_true.size(1)

    y_true_masked = y_true[mask] # (N', 9)
    y_true_mean = torch.mean(y_true_masked, dim=0, keepdim=True) # (1, 9)

    y_true_centered = y_true_masked - y_true_mean # (N', 9) - (1, 9) -> (N', 9)
    cov = torch.matmul(y_true_centered.T, y_true_centered) / (y_true_centered.size(0)-1.0) # (9, N') x (N', 9) -> (9, 9)
    cov_inv = cov.pinverse() # (9, 9)
    y_pred_expand = y_pred.unsqueeze(1).expand(B, M, N, 9) # (B, 1, N, 9) -> (B, M, N, 9)
    y_true_expand = y_true.unsqueeze(2).expand(B, M, N, 9) # (B, M, 1, 9) -> (B, M, N, 9)
    D_pairwise = y_true_expand - y_pred_expand # (B, M, N, 9)
    D_pairwise = D_pairwise[mask] # (N', N, 9)
    D_normalized = mahalanobis_norm(D_pairwise, cov_inv, y_true_mean) # (N', N)
    # Select top-K nearest neighbors between prediction and ground truth for optimization
    D_topK = torch.topk(D_normalized, k, largest=False, sorted=False, dim=0) # (k, N)
    intra_loss = D_topK.values.mean()

    y_reg1_expand = y_pred.unsqueeze(1).expand(B, N, N, 9) # (B, 1, N, 9) -> (B, N, N, 9)
    y_reg2_expand = y_pred.unsqueeze(2).expand(B, N, N, 9) # (B, N, 1, 9) -> (B, N, N, 9)
    D_reg_pairwise = (y_reg1_expand - y_reg2_expand).view(B*N,N,9) # (B, N, N, 9) -> (B*N, N, 9)
    D_reg_normalized = mahalanobis_norm(D_reg_pairwise, cov_inv, y_true_mean)  # (B*N, N)
    D_reg_topK = torch.topk(D_reg_normalized, k, largest=False, sorted=False, dim=0) # (k, N)
    inter_loss = D_reg_topK.values.mean()

    return intra_loss - 0.1 * inter_loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossWrapper, self).__init__()
        self.config = config
    def forward(self, outputs, targets, mask):
        return loss_baseline(outputs, targets, mask, self.config['loss_topk'], self.config['loss_max_pair'])
