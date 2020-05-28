import torch
import torch.nn as nn


class EulerActivation(nn.Module):

    def __init__(self):
        super(EulerActivation,self).__init__() # stateless

    def forward(self, x):
        # featurefeature_volume_batch shape: (B, M, #pitch, #yaw, 7): -> logit, xyz(3), roll(1), residual(2)
        accum = x[...,0:1] # linear
        xyz   = torch.sigmoid(x[...,1:4])
        roll  = x[...,4:5]
        pitch_residual = torch.sigmoid(x[...,5:6]) # 0~1
        yaw_residual   = torch.sigmoid(x[...,6:7]) # 0~1
        return torch.cat((accum, xyz, roll, pitch_residual, yaw_residual), -1)
