import numpy as np
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

    def __init__(self, config, activation_layer=None):
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
            .conv1d(np.prod(config['output_dim']), activation=None)
        )
        self.activation_layer = activation_layer

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
        x = x.view(x.size(0), x.size(1), *self.config['output_dim'])

        return self.activation_layer(x)
