import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Reduce
import copy

from utils.PE import PatchEmbedding
from utils.Attention import MultiHeadAttention


# 克隆N层网络
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 归一化层
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 残差链接层
class ResidualAdd(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(ResidualAdd, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# 前馈全连接层
class FeedForwardBlock(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, expansion: int = 4, dropout: float = 0.1):
        super(FeedForwardBlock, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * expansion)
        self.w_2 = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)
        self.GELU = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.GELU(self.w_1(x))))


# 单个Encoder层
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size: int, self_attn, feed_forward, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualAdd(size, dropout), 2)
        self.size = size

    def forward(self, x, mask: Tensor = None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask))
        return self.sublayer[1](x, self.feed_forward)


# 分类器
class Classifaction(nn.Module):
    # b*n*e -> b*n_classes
    def __init__(self, emb_size: int = 768, n_classes: int = 2):
        super(Classifaction, self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = Reduce('b n e -> b e', 'mean')(x)
        x = self.norm(x)
        return self.linear(x)


class VIT(nn.Module):
    def __init__(self, in_channels: int = 3, patches: int = 16, emb_size: int = 768, num_head: int = 8,
                 img_size: int = 224, depth: int = 12, n_classes: int = 2):
        super(VIT, self).__init__()
        self.depth = depth
        self.embed = PatchEmbedding(in_channels, patches, emb_size, img_size)
        self.encodelayer = EncoderLayer(emb_size, MultiHeadAttention(emb_size, num_head), FeedForwardBlock(emb_size, 4))
        self.encodes = clones(self.encodelayer, depth)
        self.cretify = Classifaction(emb_size, n_classes)

    def forward(self, x, mask: Tensor = None):
        x = self.embed(x)
        for encode in self.encodes:
            x = encode(x, mask)
        return self.cretify(x)
