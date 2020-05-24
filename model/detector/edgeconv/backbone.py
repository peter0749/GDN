import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import EdgeConv
from dgl.transform import knn_graph
import etw_pytorch_utils as pt_utils
from collections import namedtuple
from pointnet2.utils import pointnet2_utils
from ..utils import GatherPoints


class EdgeDet(nn.Module):

    def __init__(self, config):
        super(EdgeDet, self).__init__()

        self.config = config
        self.knn = config['knn']
        self.gather_points = GatherPoints()

        self.conv1 = nn.ModuleList([
            EdgeConv( 3, 32, batch_norm=True),
            EdgeConv(32, 64, batch_norm=True),
        ])

        self.conv2 = nn.ModuleList([
            EdgeConv( 64, 128, batch_norm=True),
            EdgeConv(128, 128, batch_norm=True),
            EdgeConv(128, 256, batch_norm=True),
        ])

        self.conv3 = nn.ModuleList([
            EdgeConv(256,  512, batch_norm=True),
            EdgeConv(512,  512, batch_norm=True),
            EdgeConv(512, 1024, batch_norm=True),
        ])

        self.convs = nn.ModuleList([self.conv1, self.conv2, self.conv3])

        self.FC_layer = (
            pt_utils.Seq(1024)
            .conv1d(config['n_pitch'] * config['n_yaw'] * 8, activation=None)
        )

    def forward(self, x, indices):

        h = x

        for i in range(len(self.convs)):
            for j in range(len(self.convs[i])):
                batch_size = h.size(0)
                n_points = h.size(1)
                g = knn_graph(h, self.knn[i])
                h = h.view(batch_size * n_points, -1)
                h = self.convs[i][j](g, h)
                h = F.leaky_relu(h, 0.2)
                h = h.view(batch_size, n_points, -1)
            h = self.gather_points(h, indices[i])

        x = h
        x = self.FC_layer(x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() # (B, M, n_pitch*n_yaw*8)
        x = x.view(x.size(0), x.size(1), self.config['n_pitch'], self.config['n_yaw'], 8)

        # featurefeature_volume_batch shape: (B, M, #pitch, #yaw, 8): -> logit, xyz(3), roll(2), residual(2)

        accum = x[...,0:1] # linear
        xyz   = torch.sigmoid(x[...,1:4])
        roll  = x[...,4:6]/(torch.norm(x[...,4:6], p=2, dim=-1, keepdim=True)+1e-8) # norm
        pitch_residual = torch.sigmoid(x[...,6:7]) # 0~1
        yaw_residual   = torch.sigmoid(x[...,7:8]) # 0~1

        return torch.cat((accum, xyz, roll, pitch_residual, yaw_residual), -1)
