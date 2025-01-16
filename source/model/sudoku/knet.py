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
        n,              # 神经元旋转维度
        ch=64,          # 通道数
        L=1,            # 网络深度？居然只要1层吗
        T=16,           # 每层迭代步数 
        gamma=1.0,      # kuramoto求导之后的单步步长
        J="attn",       # 链接机制，非图像识别不用convo
        use_omega=True, # 旋转频率项
        global_omg=True,# 全局旋转频率
        init_omg=0.1,   # 初始旋转频率
        learn_omg=False,# 
        nl=True,        # conv 和 归一化+激活+卷积
        heads=8,
    ):
        
        super().__init__()
        self.n = n
        self.L = L
        self.ch = ch
        self.embedding = nn.Embedding(10, ch)#数字emb到ch维

        hw = [9, 9]

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
                        ),
                        nn.Sequential(
                            ReadOutConv(ch, ch, n, 1, 1, 0),
                            #将特征张量进行降维并计算特定输出维度的特征值
                            ResBlock(FF(ch, ch, ch, 1, 1, 0)) if nl else nn.Identity(),
                            BNReLUConv2d(ch, ch, 1, 1, 0) if nl else nn.Identity(),
                            #归一化+激活+卷积
                        #self,inch,outch,kernel_size=1, stride=1,padding=0,norm=None,act=nn.ReLU(),
  
                        ),
                    ]
                )
            )

        self.out = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, 9, 1, 1, 0))
        #in_channels, out_channels, kernel_size, stride=1, padding=0,

        self.T = T
        self.gamma = torch.nn.Parameter(torch.Tensor([gamma]))#单步长是可学习参数
        self.fixed_noise = False
        self.x0 = nn.Parameter(torch.randn(1, ch, 9, 9)) #初始值代表正态分布中的采样参数，代表9*9的网格，每个网格都有ch维度的emb
        self.print_flag = True
        self.print_flag2 = True

    def feature(self, inp, is_input):
        # inp: torch.Tensor of shape [B, 9, 9, 9] the last dim repreents the digit in the one-hot representation.
        inp = convert_onehot_to_int(inp)# [B, H, W, 9]->[B, H, W]
        #在最后一个维度，将0到9的数值嵌入ch维度
        c = self.embedding(inp).permute(0, 3, 1, 2)# [B，H, W]->[B,ch, H, W]
        is_input = is_input.permute(0, 3, 1, 2)#[B, 9, 9, 1] -> [B, 1, 9, 9]
        xs = []
        es = []

        # generate random oscillatores
        if self.fixed_noise: #固定随机数种子便于复现
            n = torch.randn(
                *(c.shape), generator=torch.Generator(device="cpu").manual_seed(42)
            ).to(c.device)
            x = is_input * c + (1 - is_input) * n
        else:
            n = torch.randn_like(c)
            x = is_input * c + (1 - is_input) * n
        # c ：[B,ch, H, W]；x：[B,ch, H, W]
        # is input：x=c
        # not input：x=n

        for _, (klayer, readout) in enumerate(self.layers):
            #每次调用的其实是【klayer , nn.sequential]
            # Process x and c.
            _xs, _es = klayer(
                x,
                c,
                self.T,
                self.gamma,
            )
            xs.append(_xs)
            es.append(_es)
            c = readout(_xs[-1]) #最后一步的振荡器状态）
    
        if self.print_flag:
            self.print_flag = False
            print("xs.size()", xs.size())
            print("es.size()", es.size())
            print("each_step_size_xs_es",_xs.size(),_es.size())
        return c, xs, es

    def forward(self, c, is_input, return_xs=False, return_es=False):
        #c输入网格的初始特征
        out, xs, es = self.feature(c, is_input)
        #xs:振荡器状态的历史记录，Layers,T,B,ch,9,9
        #es：每一步的能量值，layers，T，B
        out = self.out(out).permute(0, 2, 3, 1)#输出的channel是9个
        if self.print_flag2:
            self.print_flag2 = False
            print("out.size().after 9 outchannel conv and permute(0, 2, 3, 1)", out.size())

        ret = [out]
        if return_xs:
            ret.append(xs)
        if return_es:
            ret.append(es)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
# 调用 feature 提取特征和中间状态。
# 处理最终特征，生成网格分类概率。
# 根据参数决定是否返回中间状态。
# 返回最终结果。
