# 5.2 Embedding 层
# Embedding 本质是一张可学习的查找表 W ∈ R^(num_embeddings, embedding_dim)。
# 前向: 给定整数索引 idx (任意形状)，输出 W[idx]，形状为 (*idx.shape, embedding_dim)。
#       等价于 one_hot(idx) @ W，但直接索引省去了稀疏矩阵乘法。
# 反向: dW 只在被索引到的行上累加上游梯度 (同一个 token 出现多次需要累加，用 np.add.at)。
# padding_idx: 该行初始化为 0，且不接收梯度 (常用于 <PAD> token)。
#
# 另外实现 Transformer 中的正弦位置编码 (Sinusoidal Positional Encoding):
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
import numpy as np


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx=None, seed: int = 0):
        rng = np.random.default_rng(seed)
        # 与 PyTorch 一致: 权重从 N(0, 1) 初始化
        self.weight = rng.standard_normal((num_embeddings, embedding_dim))
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.weight[padding_idx] = 0.0
        self.grad = np.zeros_like(self.weight)
        self._idx = None

    def forward(self, idx):
        idx = np.asarray(idx)
        self._idx = idx
        # 1. 直接按行索引查表
        return self.weight[idx]

    def backward(self, grad_out):
        # grad_out: (*idx.shape, embedding_dim)
        self.grad = np.zeros_like(self.weight)
        # 2. 把梯度散射回对应的行; 重复的索引需要累加，所以不能用 self.grad[idx] += ...
        np.add.at(self.grad, self._idx.reshape(-1), grad_out.reshape(-1, self.weight.shape[1]))
        # 3. padding_idx 对应的行不更新
        if self.padding_idx is not None:
            self.grad[self.padding_idx] = 0.0
        return self.grad


def sinusoidal_position_encoding(max_len: int, d_model: int):
    """
    返回形状为 (max_len, d_model) 的位置编码矩阵
    """
    pos = np.arange(max_len)[:, None]                 # (max_len, 1)
    i = np.arange(0, d_model, 2)[None, :]             # (1, d_model/2)
    div = np.power(10000.0, i / d_model)              # 10000^(2i/d_model)
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(pos / div)                   # 偶数维用 sin
    pe[:, 1::2] = np.cos(pos / div[:, : d_model // 2])  # 奇数维用 cos (兼容 d_model 为奇数)
    return pe


if __name__ == '__main__':
    import torch

    vocab, dim, pad = 10, 4, 0
    idx = np.array([[1, 2, 2, 0],
                    [5, 1, 9, 0]])  # (batch=2, seq_len=4), 0 为 <PAD>

    emb = Embedding(vocab, dim, padding_idx=pad)
    out = emb.forward(idx)
    print("Embedding output shape:", out.shape)

    # 与 PyTorch 对比前向与反向
    t_emb = torch.nn.Embedding(vocab, dim, padding_idx=pad)
    with torch.no_grad():
        t_emb.weight.copy_(torch.from_numpy(emb.weight))
    t_out = t_emb(torch.from_numpy(idx))
    grad_out = np.random.default_rng(1).standard_normal(out.shape)
    t_out.backward(torch.from_numpy(grad_out))
    emb.backward(grad_out)
    print("forward  match:", np.allclose(out, t_out.detach().numpy()))
    print("backward match:", np.allclose(emb.grad, t_emb.weight.grad.numpy()))

    # 位置编码: 词向量 + 位置向量
    pe = sinusoidal_position_encoding(max_len=idx.shape[1], d_model=dim)
    x = out + pe[None, :, :]
    print("Embedding + PE shape:", x.shape)
    print("PE:\n", np.round(pe, 4))
