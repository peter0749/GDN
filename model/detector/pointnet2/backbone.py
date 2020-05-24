import torch
import torch.nn as nn
import torch.nn.functional as F
import etw_pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2.utils import pointnet2_utils
from pointnet2.utils.pointnet2_modules import PointnetFPModule
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSGIndex as PointnetSAModuleMSG
from ..utils import GatherPoints


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

    def __init__(self, config, input_channels=0, use_xyz=True):
        super(Pointnet2MSG, self).__init__()

        self.config = config
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
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
                radii=[0.10, 0.15],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 256, 512], [c_in, 128, 256, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 512 + 512

        self.FC_layer = (
            pt_utils.Seq(c_out_2)
            .conv1d(config['n_pitch'] * config['n_yaw'] * 8, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, indices):
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

        l_xyz, l_feature = xyz, features
        for i in range(len(self.SA_modules)):
            l_xyz, l_feature = self.SA_modules[i](l_xyz, l_feature, indices[i])

        x = l_feature
        x = self.FC_layer(x).transpose(1, 2).contiguous() # (B, M, n_pitch*n_yaw*8)
        x = x.view(x.size(0), x.size(1), self.config['n_pitch'], self.config['n_yaw'], 8)

        accum = x[...,0:1] # linear
        xyz   = torch.sigmoid(x[...,1:4])
        roll  = x[...,4:6]/(torch.norm(x[...,4:6], p=2, dim=-1, keepdim=True)+1e-8) # norm
        pitch_residual = torch.sigmoid(x[...,6:7]) # 0~1
        yaw_residual   = torch.sigmoid(x[...,7:8]) # 0~1

        return torch.cat((accum, xyz, roll, pitch_residual, yaw_residual), -1)
