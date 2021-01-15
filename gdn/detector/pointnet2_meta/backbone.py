import time
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv as sparse_inv
import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2.utils import pointnet2_utils
from pointnet2.utils.pointnet2_modules import PointnetFPModule
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG

import faiss
from .knn import search_index_pytorch_fast

class ImportanceSamplingModule(nn.Module):
    def __init__(self, n_channels=3, hidden_channels=[64, 64]):
        super(ImportanceSamplingModule, self).__init__()
        self.n_channels = n_channels
        hidden_fcs = []
        channels = [n_channels,] + hidden_channels
        for i in range(1, len(channels)):
            hidden_fcs.append(nn.Linear(channels[i-1], channels[i]))
            hidden_fcs.append(nn.LeakyReLU(0.1, inplace=True))
        # (B, M, ?)
        self.hidden = nn.Sequential(*hidden_fcs)
        self.attention = nn.Sequential(
                             nn.Linear(channels[-1], 1), # (B, M, ??) -> (B, M, 1)
                             nn.Sigmoid() # (B, M, 1)
                         )
    def forward(self, x):
        f = self.hidden(x) # (B, M, ?)
        a = self.attention(f) # (B, M, 1)
        return a


class Backbone(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, config, input_channels=0, use_xyz=True):
        super(Backbone, self).__init__()

        self.config = config
        # "subsample_levels": [1024, 256, 64]
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=config['subsample_levels'][0],
                radii=[0.01, 0.03],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
                bn=self.config['use_bn']
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=config['subsample_levels'][1],
                radii=[0.025, 0.05],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=use_xyz,
                bn=self.config['use_bn']
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=config['subsample_levels'][2],
                radii=[0.10, 0.15],
                nsamples=[16, 32],
            mlps=[[c_in, 128, 256, 512], [c_in, 128, 256, 512]],
                use_xyz=use_xyz,
                bn=self.config['use_bn']
            )
        )
        c_out_2 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128, 128, 128], bn=self.config['use_bn']))
        self.FP_modules.append(PointnetFPModule(mlp=[256+c_out_0, 256, 128], bn=self.config['use_bn']))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_2+c_out_1, 256, 256], bn=self.config['use_bn']))

    def forward(self, xyz, features):
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        h = l_features[0] # (B, 128, N)
        return h.transpose(1, 2).contiguous() # (B, N, 128)

class MetaLearner(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers
        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, config, input_channels=0, use_xyz=True, activation_layer=None, return_sparsity=False):
        super(MetaLearner, self).__init__()

        self.config = config
        self.output_dim = self.config['output_dim']
        self.topk = self.config['output_topk']
        self.I_epsilon = 1e-5
        self.g_sigma2 = 0.02
        self.DPP_L_size = self.config['DPP_L_size']
        self.DPP_Y_size = self.config['DPP_Y_size']
        self.max_nproto = self.config['max_nproto']
        self.return_sparsity = return_sparsity
        self.lp_alpha = 0.99
        self.n_output_feat = np.prod(config['output_dim'])
        self.prototype_dim = 300
        self.knn = 10
        self.ivf_nlist = 8
        self.ivf_nprob = 8

        self.faiss_quantizer = faiss.IndexFlatIP(128)
        self.faiss_index = faiss.IndexIVFFlat(self.faiss_quantizer, 128, self.ivf_nlist, faiss.METRIC_L2)

        self.backbone = Backbone(self.config, input_channels=input_channels, use_xyz=use_xyz)

        self.importance_sampling = ImportanceSamplingModule(128, config['att_hidden_n'])

        self.prototype_att = ImportanceSamplingModule(128, [64, 64])
        self.prototype_l = (
            pt_utils.Seq(128)
            .conv1d(256, activation=nn.ReLU(inplace=True))
            .conv1d(self.prototype_dim, activation=nn.ReLU(inplace=True))
        )

        self.FC_layer = (
            pt_utils.Seq(self.prototype_dim)
            .conv1d(self.n_output_feat, activation=None)
        )

        self.activation_layer = activation_layer

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def make_kernel(self, D, I, sigma=1.0):
        I = I[:,1:]
        D = D[:,1:]
        D = np.exp(-D / (sigma**2.0))
        N, k = I.shape
        assert k == self.knn
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
        A = scipy.sparse.eye(Wn.shape[0]) - self.lp_alpha * Wn
        A = A.toarray().astype(np.float32)
        A = torch.from_numpy(A)
        return A

    def get_propotypes(self, support_pcs, support_masks):
        # Seed selection from support set (prototype)
        x, f = self._break_up_pc(support_pcs)
        h_support = self.backbone(x, f)
        nb, npt, nf = h_support.size() # (B, Npoints, 128)
        # (S, 128)
        support_emb = h_support[support_masks]
        subset = torch.randperm(support_emb.size(0), device=support_emb.device)[:self.max_nproto]
        support_emb = support_emb[subset]

        # Label generation from support set
        s_us = support_emb.unsqueeze(0) # (1, S, 128)
        # compute attention on which prototype to attent with
        att = self.prototype_att(s_us)[0,:] # (1, S, 1) -> (S, 1)
        s = s_us.transpose(1, 2).contiguous() # (1, 128, S)
        # compute prototype
        s = self.prototype_l(s)[0].T.contiguous() # (S, 300)
        # apply attention weights
        s = att.expand(s.size(0), self.prototype_dim) * s # (S, 300) * (S, 300)
        return support_emb, s

    def forward(self, support_pcs, support_masks, query_pc, precomputed_support):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            support_pcs: A FloatTensor with shape: (Nsupport, Npoints, 3)
            support_masks: A BooleanTensor with shape: (Nsupport, Npoints)
            query_pc: A FloatTensor with shape: (B, Npoints, 3)
        """
        xyz_query, features_query = self._break_up_pc(query_pc)
        h_query = self.backbone(xyz_query, features_query) # (B, Npoints, 128)

        if precomputed_support is None:
            support_emb, s = self.get_propotypes(support_pcs, support_masks)
        else:
            support_emb, s = precomputed_support
        support_to_return = (support_emb, s)

        # Seed selection from query set
        importance = self.importance_sampling(h_query)[...,0] # (B, N)
        importance_topk = torch.topk(importance, self.topk, dim=1, sorted=True) # (B, k)
        inds = importance_topk.indices.int() # (B, k)
        att  = importance_topk.values.clamp(min=0, max=1)  # (B, k)
        h_query_subsampled = pointnet2_utils.gather_operation(h_query.transpose(1, 2).contiguous(), inds) # (B, 128, k)
        h_query_subsampled = h_query_subsampled.transpose(1, 2).contiguous() # (B, k, 128)

        # Construct graph to match prototype & query in graph embedding
        b_query, k_query = h_query_subsampled.size()[:2]
        q_query = b_query * k_query
        n_support = support_emb.size(0)
        n_nodes = q_query + n_support
        query_set_flatten = h_query_subsampled.view(-1, 128) # (Q=B*k, 128)
        f = torch.cat((query_set_flatten, support_emb), 0) # (Q+S, 128)

        with torch.no_grad():
            c = time.time()
            # D: (Q+S, k), I: (Q+S, k)
            torch.cuda.synchronize()
            D, I = search_index_pytorch_fast(self.faiss_index, f.cpu().numpy(), self.knn + 1, nprobe=self.ivf_nprob)
            I = np.clip(I, 0, f.size(0) - 1)
            elapsed_knn = time.time() - c
            c = time.time()
            W = self.make_kernel(D, I).to(f.device) # W_inv: (Q+S, Q+S), sparse tensor
            elapsed_kernel = time.time() - c
            c = time.time()
            W_T = W.T
            W_inv = W_T.mm(W).inverse().mm(W.T)
            elapsed_inv = time.time() - c
            print("knn, kernel, inv: %.4f %.4f %.4f"%(elapsed_knn, elapsed_kernel, elapsed_inv))

        # Y: (Q+S, ndim_output)
        Y = torch.cat((torch.zeros(q_query, s.size(1), dtype=s.dtype, device=s.device), s), 0)
        #Z = torch.sparse.mm(W_inv, Y)
        Z = torch.mm(W_inv, Y)
        # Generate imtermediate results by label probagation
        x = Z[:q_query].reshape(b_query, k_query, self.prototype_dim) # (B, k, 300)
        # The final result
        x = self.FC_layer(x.transpose(1, 2).contiguous()) # (B, ?, k)
        x = x.transpose(1, 2).reshape(b_query, k_query, *self.output_dim)

        if self.return_sparsity:
            # xyz: (B, N, 3)
            ind_sub = pointnet2_utils.furthest_point_sample(xyz_query, self.DPP_L_size)
            p_sub = pointnet2_utils.gather_operation(xyz_query.transpose(1,2).contiguous(), ind_sub) # (B, 3, k)
            a_sub = pointnet2_utils.gather_operation(importance.unsqueeze(1).contiguous(), ind_sub)[:,0,:].pow(0.5) # (B, k)
            a_argsort = torch.argsort(a_sub, dim=1, descending=True).int() # (B, k)
            p_sub = pointnet2_utils.gather_operation(p_sub.contiguous(), a_argsort).contiguous() # (B, 3, k), sorted
            a_sub = pointnet2_utils.gather_operation(a_sub.unsqueeze(1).contiguous(), a_argsort)[:,0,:].contiguous() # (B, k), sorted
            A = p_sub.unsqueeze(2).expand(p_sub.size(0), 3, p_sub.size(2), p_sub.size(2)) # (B, 3, 1->k, k)
            B = p_sub.unsqueeze(3).expand(p_sub.size(0), 3, p_sub.size(2), p_sub.size(2)) # (B, 3, k, 1->k)
            g_similarity = torch.exp(-torch.norm(A-B, p=2, dim=1)/(2.0*self.g_sigma2)) # (B, k, k)
            S = g_similarity + self.I_epsilon * torch.eye(g_similarity.size(1), device=g_similarity.device, dtype=g_similarity.dtype).unsqueeze(0) # To ensure positive semidefinite: (B, k, k)
            # (B, N) -> (B, 1, N) -> (B, 1, k) -> (B, k)
            phi_i = a_sub.unsqueeze(1).expand(S.size(0), S.size(1), S.size(2)) # (B, 1->k, k)
            phi_j = a_sub.unsqueeze(2).expand(S.size(0), S.size(1), S.size(2)) # (B, k, 1->k)
            L = phi_i * S * phi_j # (B, k, k)
            e = torch.eye(L.size(1), device=L.device, dtype=L.dtype)
            detLI = torch.logdet(L + e.unsqueeze(0))
            detLY = torch.logdet(L[:,:self.DPP_Y_size, :self.DPP_Y_size])
            logDDP = detLY - detLI

            return self.activation_layer(x), inds, importance, support_to_return, -logDDP.mean()

        return self.activation_layer(x), inds, importance, support_to_return
