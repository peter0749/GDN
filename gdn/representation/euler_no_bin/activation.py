import torch
import torch.nn as nn


class EulerNoBinActivation(nn.Module):

    def __init__(self):
        super(EulerNoBinActivation, self).__init__() # stateless

    def forward(self, x):
        # featurefeature_volume_batch shape: (B, M, 8): -> logit, xyz(3), roll(2), pitch(1), yaw(1)
        accum = x[...,0:1] # linear
        xyz   = torch.sigmoid(x[...,1:4])
        roll_sin2_cos2 = x[...,4:6]/(torch.norm(x[...,4:6], p=2, dim=-1, keepdim=True)+1e-8) # sin2, cos2 on unit circle
        pitch = x[...,6:7]
        yaw   = x[...,7:8]
        return torch.cat((accum, xyz, roll_sin2_cos2, pitch, yaw), -1)
