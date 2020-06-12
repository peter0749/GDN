import torch
import torch.nn as nn


class EulerRegressionActivation(nn.Module):

    def __init__(self):
        super(EulerRegressionActivation,self).__init__() # stateless

    def forward(self, x):
        # featurefeature_volume_batch shape: (B, M, 7): -> logit, xyz(3), rpy(3)
        accum = x[...,0:1] # linear
        xyz   = torch.sigmoid(x[...,1:4])
        rpy   = x[...,4:7]
        return torch.cat((accum, xyz, rpy), -1)
