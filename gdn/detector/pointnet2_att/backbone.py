import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Variable
import etw_pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2.utils import pointnet2_utils
from pointnet2.utils.pointnet2_modules import PointnetFPModule
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG
from ..utils import GatherPoints

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

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, dtype, device, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(*shape, dtype=dtype, device=device, requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape, logits.dtype, logits.device)
    return F.softmax(y / temperature, dim=1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        with torch.no_grad():
            y_argmax = torch.argmax(y, -1, keepdim=True)
            y_hard = torch.empty(*y.size(), dtype=y.dtype, device=y.device)
            y_hard.zero_()
            y_hard.scatter_(-1, y_argmax, 1)
        y = (y_hard - y).detach() + y
    return y

class Pointnet2MSG(nn.Module):
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
        super(Pointnet2MSG, self).__init__()

        self.config = config
        # "subsample_levels": [1024, 256, 64]
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.output_dim = self.config['output_dim']
        self.topk = self.config['output_topk']
        self.I_epsilon = 1e-5
        self.g_sigma2 = 0.02
        self.DPP_L_size = self.config['DPP_L_size']
        self.DPP_Y_size = self.config['DPP_Y_size']
        self.return_sparsity = return_sparsity
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=config['subsample_levels'][0],
                radii=[0.01, 0.03],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=use_xyz,
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
            )
        )
        c_out_2 = 512 + 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256+c_out_0, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_2+c_out_1, 256, 256]))

        self.importance_sampling = ImportanceSamplingModule(128, config['att_hidden_n'])

        self.FC_layer = (
            pt_utils.Seq(128)
            .conv1d(np.prod(config['output_dim']), activation=None)
        )
        self.activation_layer = activation_layer

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def sampling(self, pointcloud, temperature=1.0):
        xyz, features = self._break_up_pc(pointcloud)

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
        ht = h.transpose(1, 2).contiguous()
        importance = self.importance_sampling(ht)[...,0] # (B, N)
        importance_topk = torch.topk(importance, self.topk, dim=1, sorted=True) # (B, k)
        inds = importance_topk.indices.int().clamp(0, importance.size(1)-1) # (B, k)
        att  = importance_topk.values  # (B, k)
        h_subsampled = pointnet2_utils.gather_operation(h, inds) # (B, 128, k)

        x = self.FC_layer(h_subsampled).transpose(1, 2).contiguous() # (B, k, n_pitch*n_yaw*8)
        x = x.view(x.size(0), x.size(1), *self.output_dim) # (B, k, n_pitch, n_yaw, 8)
        refinement = self.activation_layer(x)[...,1:] # (B, k, n_pitch, n_yaw, 7)

        # sample rotations from gumbel_softmax
        B, k = x.size(0), x.size(1)
        action = x[...,0].reshape(B, k*self.config['n_pitch']*self.config['n_yaw']) # (B, k*n_pitch*n_yaw)
        action = gumbel_softmax(action, temperature, hard=False).reshape(B, k, self.config['n_pitch'], self.config['n_yaw'], 1)

        # merge the action and refinement
        x = torch.cat((action, refinement), -1) # (B, k, n_pitch, n_yaw, 8)

        return x, inds, importance

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

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
        ht = h.transpose(1, 2).contiguous()
        importance = self.importance_sampling(ht)[...,0].clamp(0, 1) # (B, N)
        importance_topk = torch.topk(importance, self.topk, dim=1, sorted=True) # (B, k)
        inds = importance_topk.indices.int().clamp(0, importance.size(1)-1) # (B, k)
        att  = importance_topk.values  # (B, k)
        h_subsampled = pointnet2_utils.gather_operation(h, inds) # (B, 128, k)

        # h_att = att.unsqueeze(1) * h_subsampled # (B, 1, k) * (B, 128, k) -> (B, 128, k)
        # x = self.FC_layer(h_att).transpose(1, 2).contiguous() # (B, k, n_pitch*n_yaw*8)
        x = self.FC_layer(h_subsampled).transpose(1, 2).contiguous() # (B, k, n_pitch*n_yaw*8)
        x = x.view(x.size(0), x.size(1), *self.output_dim)

        if self.return_sparsity:
            # xyz: (B, N, 3)
            ind_sub = pointnet2_utils.furthest_point_sample(xyz, self.DPP_L_size).clamp(0, xyz.size(1)-1)
            p_sub = pointnet2_utils.gather_operation(xyz.transpose(1,2).contiguous(), ind_sub) # (B, 3, k)
            a_sub = pointnet2_utils.gather_operation(importance.unsqueeze(1), ind_sub)[:,0,:].pow(0.5) # (B, k)
            a_argsort = torch.argsort(a_sub, dim=1, descending=True).int() # (B, k)
            p_sub = pointnet2_utils.gather_operation(p_sub, a_argsort) # (B, 3, k), sorted
            a_sub = pointnet2_utils.gather_operation(a_sub.unsqueeze(1), a_argsort)[:,0,:] # (B, k), sorted
            A = p_sub.unsqueeze(2).expand(p_sub.size(0), 3, p_sub.size(2), p_sub.size(2)) # (B, 3, 1->k, k)
            B = p_sub.unsqueeze(3).expand(p_sub.size(0), 3, p_sub.size(2), p_sub.size(2)) # (B, 3, k, 1->k)
            g_similarity = torch.exp(-torch.norm(A-B, p=2, dim=1)/(2.0*self.g_sigma2)) # (B, k, k)
            S = g_similarity + self.I_epsilon * torch.eye(g_similarity.size(1), device=g_similarity.device, dtype=g_similarity.dtype).unsqueeze(0) # To ensure positive semidefinite: (B, k, k)
            # (B, N) -> (B, 1, N) -> (B, 1, k) -> (B, k)
            phi_i = a_sub.unsqueeze(1).expand(S.size(0), S.size(1), S.size(2)) # (B, 1->k, k)
            phi_j = a_sub.unsqueeze(2).expand(S.size(0), S.size(1), S.size(2)) # (B, k, 1->k)
            L = phi_i * S * phi_j # (B, k, k)
            if torch.all(torch.isfinite(L)):
                L = torch.where(torch.isfinite(L), L, torch.FloatTensor([1e-6]).to(L.device))
                e = torch.eye(L.size(1), device=L.device, dtype=L.dtype)
                detLI  = torch.slogdet(L + e.unsqueeze(0))
                detLY  = torch.slogdet(L[:,:self.DPP_Y_size, :self.DPP_Y_size])
                logDDP = torch.where((detLY.sign>0) & torch.isfinite(detLY.logabsdet), detLY.logabsdet, torch.zeros(1, dtype=L.dtype, device=L.device)) - \
                         torch.where((detLI.sign>0) & torch.isfinite(detLI.logabsdet), detLI.logabsdet, torch.zeros(1, dtype=L.dtype, device=L.device))
                return self.activation_layer(x), inds, importance, -logDDP.mean()
            return self.activation_layer(x), inds, importance, torch.zeros(1, dtype=L.dtype, device=L.device)
        return self.activation_layer(x), inds, importance

'''
if __name__ == '__main__':
    import json
    import sys
    from gdn.representation.euler_scene_att_ce.activation import EulerActivation
    with open(sys.argv[1], 'r') as fp:
        config = json.load(fp)
    m = Pointnet2MSG(config, activation_layer=EulerActivation(), return_sparsity=True).cuda()
    p = torch.randn((1,100,3)).cuda()
    m(p)
'''
