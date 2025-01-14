import torch
import numpy as np
import math
from einops import rearrange


def make_2dcoord(H, W, normalize=False):
    """
    Return(torch.Tensor): 2d coord values of shape [H, W, 2]
    """
    x = np.arange(H, dtype=np.float32)  # [0, H)
    y = np.arange(W, dtype=np.float32)  # [0, W)
    if normalize:
        x = x / H
        y = y / W
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    return torch.Tensor(
        np.stack([x_grid.flatten(), y_grid.flatten()], -1).reshape(H, W, 2)
    )
# H,W的网格中，每一个网格点的元素是一个二维的坐标, 默认网格尺码是float,值是从0开始的整数，不归一化



#生成一个与二维或三维空间中的坐标 coord 相关的特殊旋转矩阵
# （SO(2) 矩阵或扩展形式）。具体来说，它根据输入的坐标 coord 和
# 频率 nfreqs，生成每个坐标点在多种频率下的旋转矩阵。以下是详细的解释：
def make_SO2mats(coord, nfreqs):
    """
    Args:
    coord: [..., 2 or 3] 省略号部分是网格号码，2或3表示坐标维度
    freqs: [n_freqs, 2 or 3]  每个坐标点的频率维度。
    Return:
    mats of shape [..., n_freqs, (2 or 3), 2, 2] 每个频率生成的 2 x 2 旋转矩阵。
    """
    dim = coord.shape[-1]
    b = 10000.0
    # 对数均匀分布的频率，长度为 n_freqs，偶数频率
    freqs = torch.exp(torch.arange(0.0, 2 * nfreqs, 2) * -(math.log(b) / (2 * nfreqs)))
    grid_ths = [
        torch.einsum("...i,j->...ij", coord[..., d : d + 1], freqs).flatten(-2, -1)
        for d in range(dim) #对每一个维度（2或3维） θ d​ =coord d​ ⋅freqs
    ]
    # 有角度后，易生成标准二维旋转矩，用列表按11,12,21,22顺序存储
    _mats = [
        [
            torch.cos(grid_ths[d]),
            -torch.sin(grid_ths[d]),
            torch.sin(grid_ths[d]),
            torch.cos(grid_ths[d]),
        ]
        for d in range(dim)
    ]
    mats = [
        rearrange(torch.stack(_mats[d], -1), "... (h w)->... h w", h=2, w=2)
        for d in range(dim)
    ]
    mat = torch.stack(mats, -3)
    return mat#[..., nfreqs, dim, 2, 2]
#这种旋转矩阵可以被用作几何变换或位置编码的一部分，特别是在对称性或周期性问题中。
#coord:  torch.randn(2, 4, 3)--> (2,4,nfreq,3,2,2)
    
# GTA


@torch.jit.script
def rep_mul_x(rep, x):
    #  rep.shape=[T, F, 2, 2], x.shape=[B, H, T, F*2]
    #  时间，频率，2维旋转阵， H（高度或其他特征维度） ；F*2 个频率包含两个维度的特征。
    # 目的：将旋转矩阵 rep应用于输入张量，从而将频率相关的几何特性注入到输入特征中
    shape = x.shape
    d = rep.shape[-1] 
    #unflatten ： x 的最后一维从 [F*2] 分解为 [F, 2]   结果形状: [B, H, T, F, 2]
    return (
        # [1, 1, T, F, 2, 2] * [B, H, T, F, 2, 1] -> [B, H, T, F, 2, 2],求和，最后一个2没了
        (rep[None, None] * (x.unflatten(-1, (-1, d))[..., None, :])).sum(-1).view(shape)#再回到原来的形状
    )


@torch.jit.script
def rep_mul_qkv(rep, q, k, v):
    # rep 应用于 𝑞、𝑘、𝑣 三个输入张量
    return rep_mul_x(rep, q), rep_mul_x(rep, k), rep_mul_x(rep, v)


@torch.jit.script
def rep_mul_qk(rep, q, k):
    #类似上面的qkv，但不变v
    return rep_mul_x(rep, q), rep_mul_x(rep, k)


def embed_block_diagonal(M, n):
    """
    Embed a [h*w, d/2, 2, 2] tensor M 
    into a [h*w, d//2n, 4, 4] tensor M'
    with block diagonal structure.

    Args:
    M (torch.Tensor): Tensor of shape [h*w, d/2, 2, 2]
    n (int): Number of blocks to embed into 2nx2n structure
    函数会将 𝑛 个 2×2 矩阵嵌入到一个 4×4 矩阵中。
    Returns:
        torch.Tensor: Tensor of shape [h*w, d//2n, 4, 4]
    """
    h_w, d_half, _, _ = M.shape

    # Initialize an empty tensor for the block diagonal tensor M'
    M_prime = torch.zeros((h_w, d_half // n, 4, 4))

    # Embed M into the block diagonal structure of M_prime
    for t in range(h_w):
        for d in range(d_half // n):
            M_prime[t, d] = torch.block_diag(*[M[t, n * d + i] for i in range(n)])#每次使用了n个元素拼接
    print(M_prime.shape)
    return M_prime
#[h*w, d/2, 2, 2] --> [h*w, d//2n, 4, 4]

# e.g. : (4, 6, 2, 2) --> (4, 3, 4, 4)
# 原先的第0,0个和0,1个2*2组合成了0,0个4*4，
# 原先的第0,2个和0,3个2*2组合成了0,1个4*4……
