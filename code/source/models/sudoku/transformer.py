import torch

import torch.nn as nn
import torch.nn.functional as F
import math
from source.layers.common_layers import Attention
from source.layers.common_fns import positionalencoding2d

from source.data.datasets.sudoku.sudoku import convert_onehot_to_int


# Definition of Transformer block
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, hw=None, gta=True):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embed_dim)  # Initialize the first layer normalization
        self.attn = Attention(embed_dim, num_heads, weight="fc", gta=gta, hw=hw)  # Initialize the attention layer
        self.layernorm2 = nn.LayerNorm(embed_dim)  # Initialize the second layer normalization
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )  # Define an MLP layer with two linear mappings and an activation function

    def forward(self, src, T):
        # Repeat attention T times
        for _ in range(T):
            src2 = self.layernorm1(src)  # [B, H*W, C]
            src2 = self.attn(src2)  # [B, H*W, C]
            src = src + src2  # Residual connection [B, H*W, C]

        src2 = self.layernorm2(src)  # [B, H*W, C]
        src2 = self.mlp(src2)
        src = src + src2  # Residual connection [B, H*W, C]
        return src


# Transformer model specifically for Sudoku
class SudokuTransformer(nn.Module):
    def __init__(
        self,
        ch=64,  # Embedding dimension
        blocks=6,  # Number of Transformer blocks
        heads=4,  # Number of heads in multi-head attention
        mlp_dim=1024,  # Hidden layer dimension in MLP
        T=16,  # Number of attention iterations
        gta=False,  # Whether to use geometric-aware attention
    ):
        super().__init__()
        self.T = T

        self.embedding = nn.Embedding(10, ch)
        # If geometric-aware attention is disabled, initialize positional encoding
        if not gta:
            self.pos_embed = nn.Parameter(
                positionalencoding2d(ch, 9, 9)
                .reshape(-1, 9, 9)
                .flatten(1, 2)
                .transpose(0, 1)
            )

        self.transformer_encoder = nn.ModuleList(
            [
                TransformerBlock(
                    ch, heads, mlp_dim, 0.0, hw=(9, 9), gta=gta
                )
                for _ in range(blocks)
            ]
        )

        self.out = torch.nn.Sequential(nn.LayerNorm(ch), nn.Linear(ch, 9))

    def forward(self, x, is_input):
        B = x.size(0)
        x = convert_onehot_to_int(x)  # [B, H, W]
        x = self.embedding(x)  # [B, H, W, C]
        x = x.view(B, -1, x.shape[-1])  # [B, H*W, C]
        is_input = is_input.view(B, -1)
        # Add positional encoding
        if hasattr(self, "pos_embed"):
            x = x + self.pos_embed[None]
        # Pass through multiple Transformer blocks
        for block in self.transformer_encoder:
            x = block(x, self.T)  # [B, H*W, C]
        x = self.out(x)  # [B, H*W, 9]
        x = x.view(-1, 9, 9, 9)  # [B, 9, 9, 9]
        return x
