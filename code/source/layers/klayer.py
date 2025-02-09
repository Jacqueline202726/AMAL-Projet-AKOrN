import torch
import torch.nn as nn
# from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils import weight_norm

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
    """
    Defines the Omega layer, which simulates the natural frequency characteristics of an oscillator.
    """

    def __init__(self, n, ch, init_omg=0.1, global_omg=False, learn_omg=True):
        super().__init__()
        self.n = n  # Dimension of each oscillator
        self.ch = ch  # Number of channels
        self.global_omg = global_omg  # Whether to share omega globally

        if not learn_omg:
            print("Not learning omega")

        if n % 2 != 0:
            # n is odd
            raise NotImplementedError
        else:
            # n is even (initialize omega parameter, supports learning or freezing)
            if global_omg:  # In global sharing mode, all oscillators share the same omega parameter
                self.omg_param = nn.Parameter(
                    init_omg * (1 / np.sqrt(2)) * torch.ones(2), requires_grad=learn_omg
                )
            else:  # In non-global sharing mode, each group of (ch // 2) oscillators has its own omega parameter
                self.omg_param = nn.Parameter(
                    init_omg * (1 / np.sqrt(2)) * torch.ones(ch // 2, 2),
                    requires_grad=learn_omg,
                )

    def forward(self, x):
        """
        Compute the transformation of omega with the input tensor.
        """
        _x = reshape(x, 2)  # Adjusts input tensor to [B, T, 2] or similar, representing each oscillator in 2D space
        if self.global_omg:
            omg = torch.linalg.norm(self.omg_param).repeat(_x.shape[1])  # Global omega (broadcasted to [T] size)
        else:
            omg = torch.linalg.norm(self.omg_param, dim=1)  # Assign omega per channel
        omg = omg[None]
        for _ in range(_x.ndim - 3):  # Adjust dimensions to be compatible with input shape
            omg = omg.unsqueeze(-1)
        # Compute the rotated tensor using the rotation formula:
        # x_new = omg * y
        # y_new = -omg * x
        omg_x = torch.stack([omg * _x[:, :, 1], -omg * _x[:, :, 0]], dim=2)
        omg_x = reshape_back(omg_x)  # Restore original tensor shape
        return omg_x


class KLayer(nn.Module):  # Kuramoto layer

    def __init__(
        self,
        n,  # Rotation dimension of each neuron
        ch,  # Number of channels
        J="conv",  # Connectivity type of Kuramoto layer
        c_norm="gn",  # Normalization method for c
        use_omega=False,  # Whether to enable the omega term
        init_omg=1.0,  # Initial value of omega
        ksize=3,  # Kernel size
        gta=False,  # Whether to use geometric attention
        hw=None,  # Height and width of the input
        global_omg=False,  # Whether omega is globally shared
        heads=8,  # Number of attention heads
        learn_omg=True,  # Whether omega is learnable
        apply_proj=True,  # Whether to apply projection
    ):
        # Connectivity is either 'conv' or 'ca'
        super().__init__()
        assert (ch % n) == 0  # Ensure the number of channels is divisible by n
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

        # Define connectivity type (convolution or attention)
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
        else:
            raise NotImplementedError

        # Define normalization method
        if c_norm == "gn":
            self.c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            self.c_norm = ScaleAndBias(ch, token_input=False)
        elif c_norm is None or c_norm == "none":
            self.c_norm = nn.Identity()
        else:
            raise NotImplementedError

    def project(self, y, x):
        """
        Projection operation: projects the updated vector onto the tangent space.
        """
        sim = x * y  # Similarity between update and current state
        yxx = torch.sum(sim, 2, keepdim=True) * x
        return y - yxx, sim

    def kupdate(self, x: torch.Tensor, c: torch.Tensor = None):
        """
        Perform Kuramoto update.
        """
        # Compute  \sum_j[J_ij x_j]
        _y = self.connectivity(x)  # Compute connectivity-based update
        # Add bias c
        y = _y + c  # Add bias term

        if hasattr(self, "omg"):
            omg_x = self.omg(x)
        else:
            omg_x = torch.zeros_like(x)

        # Reshape tensors for further operations
        y = reshape(y, self.n)
        x = reshape(x, self.n)

        # Project y onto the tangent space
        if self.apply_proj:
            y_yxx, sim = self.project(y, x)
        else:
            y_yxx = y
            sim = y * x

        dxdt = omg_x + reshape_back(y_yxx)  # Update x
        sim = reshape_back(sim)

        return dxdt, sim

    def forward(self, x: torch.Tensor, c: torch.Tensor, T: int, gamma):
        """
        Perform T-step Kuramoto updates.
        """
        # x.shape = c.shape = [B, C,...] or [B, T, C]
        xs, es = [], []
        c = self.c_norm(c)  # Normalize conditioning input
        x = normalize(x, self.n)  # Normalize input state
        es.append(torch.zeros(x.shape[0]).to(x.device))
        # Iterate Kuramoto updates with conditioning input c
        for t in range(T):
            dxdt, _sim = self.kupdate(x, c)
            x = normalize(x + gamma * dxdt, self.n)  # Update state
            xs.append(x)
            es.append((-_sim).reshape(x.shape[0], -1).sum(-1))

        return xs, es
