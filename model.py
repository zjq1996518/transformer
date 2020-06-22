import torch
from torch.nn import Module, Linear, LayerNorm, LeakyReLU
import torch.nn.functional as F
import numpy as np


def mask(length):
    """
    生成mask矩阵的方法
    :param length: [length1, length2, length3, ...]
    :return:
    """
    origin_tensor = torch.ones(length[0].item() + 1, length[0].item() + 1).cuda().detach()
    triu_tensor = torch.triu(origin_tensor, diagonal=0).detach()
    mask_tensor = ((torch.index_select(triu_tensor, dim=0, index=length)[:, :-1] - 1) * -1).detach()

    return mask_tensor


class MultiHeadAttention(Module):

    def __init__(self, input_size, key_size, heads):

        super().__init__()

        self.input_size = input_size
        self.key_size = key_size
        self.heads = heads

        # bias设置为false 是为了防止padding为0的部分变为非0
        self.q_w = Linear(input_size, key_size*heads, bias=False)
        self.k_w = Linear(input_size, key_size*heads, bias=False)
        self.v_w = Linear(input_size, input_size*heads, bias=False)
        # 融合多头信息
        self.linear = Linear(input_size*heads, input_size)

        self.layer_norm = LayerNorm(input_size, eps=1e-6)

    def forward(self, x):

        # [B, T, input_size]
        shape = x.shape

        # q, k => [B, T, H*key_size]  v=> [B, T, H*input_size]
        k = self.k_w(x)
        q = self.q_w(x)
        v = self.v_w(x)

        # q, k => [B, T, H, key_size]  v=> [B, T, H, input_size]
        q, k, v = [vector.reshape(*shape[:2], self.heads, -1) for vector in [q, k, v]]

        # q, k => [H, B, T, key_size]  v=> [H, B, T, input_size]
        q, k, v = [vector.permute((2, 0, 1, 3)) for vector in [q, k, v]]

        # # q, k => [H*B, T, key_size]  v=> [H*B, T, input_size]
        # q, k, v = [vector.reshape(vector.shape[0]*vector.shape[1], *vector.shape[2:]) for vector in [q, k, v]]

        # att_score => [H, B, Tq, Tk]
        # ScaledDotProductAttention
        att_score = torch.matmul(q, k.transpose(-1, -2)) / self.key_size**0.5
        att_score = att_score.masked_fill(att_score == 0, -1e9)
        att_score = F.softmax(att_score, dim=-1)
        # att_score => [H, B, T, input_size]
        att_score = torch.matmul(att_score, v)

        # att_score => [B, T, H, input_size]
        att_score = att_score.permute((1, 2, 0, 3))

        # att_score => [B, T, H*input_size]
        att_score = att_score.reshape(*shape[:2], self.heads*self.input_size)

        att_score = self.linear(att_score)

        # 残差
        att_score = self.layer_norm(att_score + x)

        return att_score


class PositionalEncoding(Module):

    def __init__(self, hidden, n_position=90):
        super(PositionalEncoding, self).__init__()

        self.hidden = hidden
        self.register_buffer('pos_vector', self.get_pos_vector(n_position))

    def get_vector(self, position):
        return [position / np.power(10000, 2 * (j // 2) / self.hidden) for j in range(self.hidden)]

    def get_pos_vector(self, n):

        vector = np.array([self.get_vector(i) for i in range(n)])
        vector[:, 0::2] = np.sin(vector[:, 0::2])
        vector[:, 1::2] = np.cos(vector[:, 1::2])

        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_vector[:, :x.shape[1]].detach()


class TransformerLayer(Module):
    """
    这里只实现了transformer的编码器
    """
    def __init__(self, input_size, key_size, heads):

        super().__init__()
        self.attn = MultiHeadAttention(input_size, key_size, heads)

        # bias设置为false 是为了防止padding为0的部分变为非0
        self.linear1 = Linear(input_size, input_size, bias=False)
        self.active = LeakyReLU()
        self.linear2 = Linear(input_size, input_size, bias=False)
        self.layer_norm = LayerNorm(input_size, eps=1e-6)

    def forward(self, x, mask_tensor):
        """
        :param x: 输入 shape [B x T x H]
        :param mask_tensor: shape [B x T x H]
        编码器的mask_tensor矩阵只需要对长度padding部分mask即可
        :return:
        """
        x = x.masked_fill(mask_tensor.reshape(*mask_tensor.shape, 1) == 0, 0)
        x = self.attn(x)
        y = self.linear2(self.active(self.linear1(x)))

        y = x + y
        y = self.layer_norm(y)

        return y
