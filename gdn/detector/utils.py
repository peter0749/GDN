import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from pointnet2.utils import pointnet2_utils
from .knn import search_index_pytorch_fast


def make_kernel(D, I, sigma=1.0, lp_alpha=0.99):
    I = I[:,1:]
    D = D[:,1:]
    D = np.exp(-D / (sigma**2.0))
    N, k = I.shape
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx,(k,1)).T
    W = csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T # make it symmetric
    W = W - scipy.sparse.diags(W.diagonal()) # remove self loop
    S = W.sum(axis = 1)
    S[S==0] = 1
    D = np.array(1./ np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D
    A = scipy.sparse.eye(Wn.shape[0]) - lp_alpha * Wn
    A = A.toarray().astype(np.float32)
    A = torch.from_numpy(A)
    return A


def make_propagation_kernel(h, n_neighbors=5):
    '''
    h: (B, 128, N)
    '''
    B, C, N = h.size()
    faiss_index = faiss.IndexFlatL2(C)
    h_cpu = np.transpose(h.cpu().numpy(), (0, 2, 1))
    Ws = []
    Ws_inv = []
    with torch.no_grad():
        for b in range(B):
            D, I = search_index_pytorch_fast(faiss_index, h_cpu[b], n_neighbors+1)
            I = np.clip(I, 0, N - 1)
            W = make_kernel(D, I)
            Ws.append(W)
        for b in range(B):
            W = Ws[b]
            W_T = W.T
            W_inv = W_T.mm(W).inverse().mm(W.T)
            Ws_inv.append(W_inv.unsqueeze(0))
        del Ws
        return torch.cat(Ws_inv, 0).to(h.device)

def freeze_model(m):
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_model(m):
    for param in m.parameters():
        param.requires_grad = True


class GatherPoints(nn.Module):
    def __init__(self):
        super(GatherPoints, self).__init__()
    def forward(self, xyz, point_indices):
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        return pointnet2_utils.gather_operation(xyz_flipped, point_indices).transpose(1, 2).contiguous()


def rotation_tensor(roll, pitch, yaw):
    r = roll.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
    p = pitch.unsqueeze(-1).unsqueeze(-1)
    y = yaw.unsqueeze(-1).unsqueeze(-1)
    ones  = torch.ones_like(r)
    zeros = torch.zeros_like(r)
    r_sin = r.sin()
    r_cos = r.cos()
    p_sin = p.sin()
    p_cos = p.cos()
    y_sin = y.sin()
    y_cos = y.cos()
    Rx = torch.cat((
        torch.cat((ones, zeros, zeros), 1),
        torch.cat((zeros, r_cos, r_sin), 1),
        torch.cat((zeros, -r_sin, r_cos), 1)
    ), 2)
    Ry = torch.cat((
        torch.cat((p_cos, zeros, -p_sin), 1),
        torch.cat((zeros, ones, zeros), 1),
        torch.cat((p_sin, zeros, p_cos), 1)
    ), 2)
    Rz = torch.cat((
        torch.cat((y_cos, y_sin, zeros), 1),
        torch.cat((-y_sin, y_cos, zeros), 1),
        torch.cat((zeros, zeros, ones), 1)
    ), 2)
    return torch.bmm(Rz, torch.bmm(Ry, Rx))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
