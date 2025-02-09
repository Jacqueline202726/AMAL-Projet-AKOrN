import torch
import torch.nn as nn
from source.layers.klayer import (
    KLayer,
)
from source.layers.common_layers import (
    ReadOutConv,
    BNReLUConv2d,
    FF,
    ResBlock,
)
from source.layers.common_fns import positionalencoding2d


from source.data.datasets.sudoku.sudoku import convert_onehot_to_int


class SudokuAKOrN(nn.Module):

    def __init__(
        self,
        n,  # Rotation dimension of each neuron
        ch=64,  # Number of channels
        L=1,  # Number of network layers
        T=16,  # Number of Kuramoto update iterations per layer
        gamma=1.0,  # Scaling factor for Kuramoto update step size
        J="attn",  # Connectivity type of the Kuramoto layer
        use_omega=True,  # Whether to enable the omega term
        global_omg=True,  # Whether omega is globally shared
        init_omg=0.1,  # Initial value of omega
        learn_omg=False,  # Whether omega is learnable
        nl=True,  # Whether to use non-linear activation
        heads=8,  # Number of attention heads
    ):
        super().__init__()
        self.n = n
        self.L = L
        self.ch = ch
        self.embedding = nn.Embedding(10, ch)  # Embeds Sudoku digits into feature vectors of dimension ch

        hw = [9, 9]  # Grid size

        self.layers = nn.ModuleList()
        for l in range(self.L):
            self.layers.append(
                nn.ModuleList(
                    [
                        KLayer(
                            n,
                            ch,
                            J,
                            use_omega=use_omega,
                            c_norm=None,
                            hw=hw,
                            global_omg=global_omg,
                            init_omg=init_omg,
                            heads=heads,
                            learn_omg=learn_omg,
                            gta=True,
                        ),  # Kuramoto oscillator layer
                        nn.Sequential(
                            ReadOutConv(ch, ch, n, 1, 1, 0),
                            ResBlock(FF(ch, ch, ch, 1, 1, 0)) if nl else nn.Identity(),
                            BNReLUConv2d(ch, ch, 1, 1, 0) if nl else nn.Identity(),
                        ),  # Post-processing module
                    ]
                )
            )

        self.out = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, 9, 1, 1, 0))  # Output layer: Maps features to 9 channels, representing 9 possible values per cell

        self.T = T
        self.gamma = torch.nn.Parameter(torch.Tensor([gamma]))  # Learnable step size parameter
        self.fixed_noise = False  # Controls whether noise is fixed

    def feature(self, inp, is_input):
        # inp: torch.Tensor of shape [B, 9, 9, 9], where the last dimension represents the digit in one-hot encoding.
        inp = convert_onehot_to_int(inp)  # Convert input from one-hot encoding to integers, shape [B, H, W]
        c = self.embedding(inp).permute(0, 3, 1, 2)  # Use embedding layer to convert integers into feature vectors and reshape to [B, ch, 9, 9] (channel dimension ch placed in second dimension for convolution compatibility)
        is_input = is_input.permute(0, 3, 1, 2)  # [B, 1, 9, 9]
        xs = []  # Store x states for each layer
        es = []  # Store energy changes for each layer

        # Generate random oscillators # Initialize random noise
        if self.fixed_noise:
            n = torch.randn(
                *(c.shape), generator=torch.Generator(device="cpu").manual_seed(42)
            ).to(c.device)
            x = is_input * c + (1 - is_input) * n  # Initial x state composed of input and noise
        else:
            n = torch.randn_like(c)
            x = is_input * c + (1 - is_input) * n  # Initial x state composed of input and noise

        for _, (klayer, readout) in enumerate(self.layers):
            # Process x and c.
            _xs, _es = klayer(
                x,
                c,
                self.T,
                self.gamma,
            )  # Each x_t has shape [B, ch, 9, 9]; each e_t has shape [B]
            xs.append(_xs)
            es.append(_es)
            
            x = _xs[-1]
            c = readout(x)  # [B, ch, 9, 9]

        return c, xs, es  # Return final feature c, all x states, and energy changes

    def forward(self, c, is_input, return_xs=False, return_es=False):
        out, xs, es = self.feature(c, is_input)  # xs: History of oscillator states, shape [Layers, T, B, ch, 9, 9]; es: Energy values per step, shape [Layers, T, B]
        out = self.out(out).permute(0, 2, 3, 1)  # [B, 9, 9, 9]
        ret = [out]
        if return_xs:
            ret.append(xs)
        if return_es:
            ret.append(es)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
