import torch
import torch.nn as nn
import torch.nn.functional as F


def cvtD6SO3(d6_):
    '''
    d6_: (N, 6) -- 6D representation of rotation
    ---
    returns:
        R3x3: (N, 3, 3) -- rotation matrics
    '''
    N = d6_.size(0)
    d6 = d6_.view(N, 3, 2)
    a1 = d6[:,:,0] # (N, 3)
    a2 = d6[:,:,1] # (N, 3)
    b1 = F.normalize(a1, p=2, dim=1) # (N, 3)
    dot_b1_a2 = torch.bmm(b1.view(N,1,-1), a2.view(N,-1,1)).view(N,1) # (N, 1, 3) * (N, 3, 1) -> (N, 1)
    b2 = F.normalize(a2-dot_b1_a2*b1, p=2, dim=1) # (N, 3)
    b3 = torch.cross(b1, b2, dim=1) # (N, 3)
    return torch.stack([b1, b2, b3], dim=1).permute(0,2,1) # (N, 3, 3)

class S4GActivation(nn.Module):

    def __init__(self):
        super(S4GActivation,self).__init__() # stateless

    def forward(self, x):
        # featurefeature_volume_batch shape: (B, N, 10): -> logit(1), rot_6d(6), translation(3)
        logit  = x[...,0:1]
        rot_6d = x[...,1:7] # (B, N, 6)
        trans  = x[...,7:10]
        B = x.size(0)
        N = x.size(1)
        rot_9d = cvtD6SO3(rot_6d.view(-1, 6)).reshape(B, N, 9) # (B, N, 9)
        return torch.cat((logit, rot_9d, trans), 2) # (B, N, 13)
