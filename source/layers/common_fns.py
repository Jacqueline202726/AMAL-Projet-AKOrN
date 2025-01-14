import torch
import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    # 如果d_model不是4的倍数，则抛出异常
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    # 初始化一个全为0的矩阵
    pe = torch.zeros(d_model, height, width)
    # 每个维度使用d_model的一半
    d_model = int(d_model / 2)
    # 计算除数项
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    # 计算宽度位置
    pos_w = torch.arange(0., width).unsqueeze(1)
    # 计算高度位置
    pos_h = torch.arange(0., height).unsqueeze(1)
    # 计算sin和cos位置编码
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

#对于偶数维度（0到 d_model-1），使用宽度索引 pos_w 生成正弦编码，并沿高度方向重复。
#对于奇数维度（d_model 到 2*d_model-1），使用宽度索引 pos_w 生成余弦编码，并沿高度方向重复。
#对于 d_model 到 2*d_model-1 之间的维度，使用高度索引 pos_h 生成正弦编码，并沿宽度方向重复。
#对于 2*d_model 到 3*d_model-1 之间的维度，使用高度索引 pos_h 生成余弦编码，并沿宽度方向重复。