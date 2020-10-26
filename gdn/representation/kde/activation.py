import torch
import torch.nn as nn
import torch.nn.functional as F

def normalizeD6(d6_):
    '''
    d6_: (N, 6) -- 6D representation of rotation
    ---
    returns:
        normalized d6
    '''
    N = d6_.size(0)
    d6 = d6_.view(N, 3, 2)
    a1 = d6[:,:,0] # (N, 3)
    a2 = d6[:,:,1] # (N, 3)
    b1 = F.normalize(a1, p=2, dim=1) # (N, 3)
    dot_b1_a2 = torch.bmm(b1.view(N,1,-1), a2.view(N,-1,1)).view(N,1) # (N, 1, 3) * (N, 3, 1) -> (N, 1)
    b2 = F.normalize(a2-dot_b1_a2*b1, p=2, dim=1) # (N, 3)
    return torch.stack([b1, b2], dim=1).permute(0,2,1) # (N, 3, 2)

def cvtD6SO3(d6):
    '''
    d6: (N, 3, 2) -- 6D representation of rotation
    ---
    returns:
        R3x3: (N, 3, 3) -- rotation matrics
    '''
    b3 = torch.cross(b1[:,:,0:1], b2[:,:,1:2], dim=1) # (N, 3, 1)
    return torch.cat([d6, b3], dim=2) # (N, 3, 3)

class KDEActivation(nn.Module):

    def __init__(self, return_mtx=False):
        self.return_mtx = return_mtx
        super(KDEActivation,self).__init__() # stateless

    def forward(self, x):
        # featurefeature_volume_batch shape: (B, N, 9): -> rot_6d(6), translation(3)
        B = x.size(0)
        N = x.size(1)
        rot_6d = normalizeD6(x[...,:6]) # (B*N, 3, 2)
        trans  = x[...,6:9] # (B, N, 3)
        if self.return_mtx:
            rot_9d = cvtD6SO3(rot_6d).reshape(B, N, 3, 3) # (B, N, 3, 3)
            trans = trans.reshape(B, N, 3, 1) # (B, N, 3, 1)
            return torch.cat((rot_9d, trans), 3) # (B, N, 3, 4)
        return torch.cat((rot_6d.reshape(B, N, 6), trans), 2) # (B, N, 9)
