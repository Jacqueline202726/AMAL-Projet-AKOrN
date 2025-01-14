import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

import numpy as np

from source.layers.common_layers import (
    ScaleAndBias,
    Attention,
)

from source.layers.kutils import (
    reshape,
    reshape_back,
    normalize,
)

from einops.layers.torch import Rearrange


class OmegaLayer(nn.Module):
    #实现振荡器动态行为

    def __init__(self, n, ch, init_omg=0.1, global_omg=False, learn_omg=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.global_omg = global_omg #决定是否全局共享神经元

        if not learn_omg:
            print("Not learning omega")

        if n % 2 != 0:
            # n is odd
            raise NotImplementedError
        else:
            # n is even 是旋转维度，必须为偶数
            if global_omg:
                self.omg_param = nn.Parameter(
                    init_omg * (1 / np.sqrt(2)) * torch.ones(2), requires_grad=learn_omg
                )
            else:#ch//2表示对每一对通道进行分组处理，给出独立的omega参数
                self.omg_param = nn.Parameter(
                    init_omg * (1 / np.sqrt(2)) * torch.ones(ch // 2, 2),
                    requires_grad=learn_omg,
                )

    def forward(self, x):
        _x = reshape(x, 2) #转换为[B,c/2,...,2]
        if self.global_omg:
            omg = torch.linalg.norm(self.omg_param).repeat(_x.shape[1])
        else:
            omg = torch.linalg.norm(self.omg_param, dim=1)
        omg = omg[None] #最前方增加一个维度，【1，……】
        for _ in range(_x.ndim - 3):
            omg = omg.unsqueeze(-1) #使其在后续操作中可以正确广播。
        omg_x = torch.stack([omg * _x[:, :, 1], -omg * _x[:, :, 0]], dim=2)
        omg_x = reshape_back(omg_x)
        return omg_x#很奇怪，但是最后是给出了一个乘积的结果，在具体数独的时候看看怎么搞


class KLayer(nn.Module):  # Kuramoto layer

    def __init__(
        self,
        n, #旋转维度（振荡器的维度），决定每对通道如何分组
        ch, #输入的通道数。
        J="conv", #连接方式，卷积or注意力
        c_norm="gn", # gn（GroupNorm
        use_omega=False,
        init_omg=1.0,
        ksize=3,  #卷积核大小
        gta=False, #GTA 坐标嵌入
        hw=None, #输入特征的高度和宽度。
        global_omg=False, #决定是否全局共享神经元
        heads=8, 
        learn_omg=True,
        apply_proj=True,#限定在球面
    ):
        # connnectivity is either 'conv' or 'ca'
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch
        self.use_omega = use_omega
        self.global_omg = global_omg
        self.apply_proj = apply_proj

        self.omg = (
            OmegaLayer(n, ch, init_omg, global_omg, learn_omg)
            if self.use_omega
            else nn.Identity()
        )

        if J == "conv":
            self.connectivity = nn.Conv2d(ch, ch, ksize, 1, ksize // 2)
            self.x_type = "image"
        elif J == "attn":
            self.connectivity = Attention(
                ch,
                heads=heads,
                weight="conv",
                kernel_size=1,
                stride=1,
                padding=0,
                gta=gta,
                hw=hw,
            )
            self.x_type = "image"
        else:   #默认都是conv，如果数独也是conv那就不用attention了
            raise NotImplementedError

        if c_norm == "gn":
            self.c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            self.c_norm = ScaleAndBias(ch, token_input=False)
        elif c_norm is None or c_norm == "none":
            self.c_norm = nn.Identity()
        else:
            raise NotImplementedError

    def project(self, y, x):
        # [B,C,n],逐元素相乘，结果不变
        sim = x * y  # similarity between update and current state
        yxx = torch.sum(sim, 2, keepdim=True) * x  # [B,C,1]*[B,C,n]
        #投影长度*x，得到半径方向的部分，相减得到切线
        return y - yxx, sim
    #核心函数，kupdate
    def kupdate(self, x: torch.Tensor, c: torch.Tensor = None):
        # compute  \sum_j[J_ij x_j]
        _y = self.connectivity(x)#卷积或者多头注意力处理

        # add bias c.
        y = _y + c #输入的时候是X和C一起输出

        if hasattr(self, "omg"):
            omg_x = self.omg(x)
        else:
            omg_x = torch.zeros_like(x)

        y = reshape(y, self.n)
        x = reshape(x, self.n)

        # project y onto the tangent space
        if self.apply_proj:
            y_yxx, sim = self.project(y, x)
        else:
            y_yxx = y
            sim = y * x

        dxdt = omg_x + reshape_back(y_yxx) #等式右侧！
        sim = reshape_back(sim)

        return dxdt, sim

    def forward(self, x: torch.Tensor, c: torch.Tensor, T: int, gamma):
        # x.shape = c.shape = [B, C,...] or [B, T, C]
        xs, es = [], []#x：每个时间步的状态序列，es：每个时间步的能量
        c = self.c_norm(c)#归一化C
        x = normalize(x, self.n) #单位球面归一化
        es.append(torch.zeros(x.shape[0]).to(x.device))
        # Iterate kuramoto update with condition c
        for t in range(T):
            dxdt, _sim = self.kupdate(x, c)
            x = normalize(x + gamma * dxdt, self.n)
            xs.append(x)
            es.append((-_sim).reshape(x.shape[0], -1).sum(-1))

        return xs, es
