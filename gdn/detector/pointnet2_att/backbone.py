import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        h = l_features[0] # (B, 128, M)
        ht = h.transpose(1, 2).contiguous()
        importance = self.importance_sampling(ht)[...,0] # (B, M)
        importance_topk = torch.topk(importance, self.topk, dim=1) # (B, k)
        inds = importance_topk.indices.int() # (B, k)
        att  = importance_topk.values  # (B, k)
        h_subsampled = pointnet2_utils.gather_operation(h, inds) # (B, 128, k)

        h_att = att.unsqueeze(1) * h_subsampled # (B, 1, k) * (B, 128, k) -> (B, 128, k)
        x = self.FC_layer(h_att).transpose(1, 2).contiguous() # (B, k, n_pitch*n_yaw*8)
        x = x.view(x.size(0), x.size(1), *self.output_dim)

        if self.return_sparsity:
            w = self.importance_sampling.hidden[-2].weight
            # w: (out_features, in_features) -> each output neuron
            #                                   has $in_features groups
            #                                   and output 1 scalar
            w_norm = (w*w).sum(1, keepdim=True).pow(0.5) # (out_features, 1)
            w_n = w / w_norm # Normalized W: (out_features, in_features) / (out_features, 1)
            feature_correlation = torch.mm(w_n, w_n.T) # (out_features, out_features)
            diversity = (1.0 - feature_correlation * (1.0 - torch.eye(w.size(0), w.size(0), dtype=w.dtype, device=w.device))).sum()
            return self.activation_layer(x), inds, importance, diversity

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
