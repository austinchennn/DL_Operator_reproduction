import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # 每个 KV 头被多少个 Q 头共享

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(b, s, self.n_heads, self.head_dim)
        k = k.view(b, s, self.n_kv_heads, self.head_dim)
        v = v.view(b, s, self.n_kv_heads, self.head_dim)

        # 重复 KV 头以匹配 Q 的数量 (Key Repeat)
        # 这种方式比 MHA 节省显存，比 MQA 精度更高
        k = torch.repeat_interleave(k, repeats=self.n_rep, dim=2)
        v = torch.repeat_interleave(v, repeats=self.n_rep, dim=2)

        # 随后进行标准 Scaled Dot-Product Attention
        # ... (此处省略 Attention 计算代码)
        return self.wo(output)