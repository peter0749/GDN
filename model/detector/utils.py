import torch
import torch.nn as nn
from pointnet2.utils import pointnet2_utils


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
