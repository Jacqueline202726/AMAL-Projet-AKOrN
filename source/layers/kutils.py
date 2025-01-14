from sympy import prod
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import einops


def reshape(x: torch.Tensor, n: int):
    if x.ndim == 3:  # x.shape = ([B, T, C ]) Batch, Timesteps, Channels
        #[B, T, C ]-> [B, C, T] ->[B, C'=C // n   , n, T]，
        return x.transpose(1, 2).unflatten(1, (-1, n)) 
    else:  # x.shape = ([B, C, ..., ])
        return x.unflatten(1, (-1, n)) #all change channel 


def reshape_back(x):
    if x.ndim == 4:  # Tokens [B, C', n, T]
        return x.flatten(1, 2).transpose(1, 2) # [B, C', n, T] -> [B, T, C]
    else:
        return x.flatten(1, 2) #改变第二个维度


def _l2normalize(x): #dim =0,1,2 看起来像是对channel进行正则化
    return torch.nn.functional.normalize(x, dim=2)


def norm(n, x, dim=2, keepdim=True): #第 0,1,2,3 维度
    #计算在指定维度的 L2 范数，支持对分块后的张量操作。    
    return torch.linalg.norm(reshape(x, n), dim=dim, keepdim=keepdim)


def normalize(x: torch.Tensor, n):
    #原先的维度是[B, C', n, T]，reshape后是[B, T, C]，对channel正则化
    x = reshape(x, n) #[B, C', n, T] -> [B, T, C]
    x = _l2normalize(x)  #
    x = reshape_back(x)
    return x


class Normalize(nn.Module):

    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return normalize(self.n, x)
# example   
# normalize_layer = Normalize(n=2)
# output = normalize_layer(input_tensor)

# currently not used
def compute_exponential_map(n, x, dxdt, reshaped_inputs=False):
    if not reshaped_inputs:
        dxdt = reshape(dxdt, n)
        x = reshape(x, n)
    norm = torch.linalg.norm(dxdt, dim=2, keepdim=True)
    norm = torch.clip(norm, 0, math.pi)
    nx = torch.cos(norm) * x + torch.sin(norm) * (dxdt / (norm + 1e-5))
    if not reshaped_inputs:
        nx = reshape_back(nx)
    return nx

