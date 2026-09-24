# 5.3 循环层: RNN / LSTM / GRU
# 输入 x: (batch, seq_len, input_size)，初始隐状态 h0: (batch, hidden_size)
# 输出 output: (batch, seq_len, hidden_size) 每个时间步的隐状态，以及最后时刻的状态。
# 权重布局与 PyTorch 保持一致 (W_ih: (G*H, I), W_hh: (G*H, H))，G 为门的个数，方便对拍验证。
#
# RNN : h_t = tanh(x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh)
# LSTM: [i, f, g, o] = x_t W_ih^T + b_ih + h_{t-1} W_hh^T + b_hh   (按顺序切成 4 份)
#       i, f, o = sigmoid(.)，g = tanh(.)
#       c_t = f * c_{t-1} + i * g
#       h_t = o * tanh(c_t)
# GRU : [r, z, n] 三个门，注意 n 中重置门 r 作用在 (h W_hn^T + b_hn) 上
#       r = sigmoid(x W_ir^T + b_ir + h W_hr^T + b_hr)
#       z = sigmoid(x W_iz^T + b_iz + h W_hz^T + b_hz)
#       n = tanh(x W_in^T + b_in + r * (h W_hn^T + b_hn))
#       h_t = (1 - z) * n + z * h_{t-1}
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class _RecurrentBase:
    num_gates = 1

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        G, H = self.num_gates, hidden_size
        # 与 PyTorch 一致: U(-1/sqrt(H), 1/sqrt(H)) 初始化
        k = 1.0 / np.sqrt(H)
        self.hidden_size = H
        self.W_ih = rng.uniform(-k, k, (G * H, input_size))
        self.W_hh = rng.uniform(-k, k, (G * H, H))
        self.b_ih = rng.uniform(-k, k, G * H)
        self.b_hh = rng.uniform(-k, k, G * H)


class RNN(_RecurrentBase):
    num_gates = 1

    def forward(self, x, h0=None):
        B, T, _ = x.shape
        h = np.zeros((B, self.hidden_size)) if h0 is None else h0
        outputs = []
        for t in range(T):
            h = np.tanh(x[:, t] @ self.W_ih.T + self.b_ih + h @ self.W_hh.T + self.b_hh)
            outputs.append(h)
        return np.stack(outputs, axis=1), h


class LSTM(_RecurrentBase):
    num_gates = 4

    def forward(self, x, state=None):
        B, T, _ = x.shape
        H = self.hidden_size
        if state is None:
            h, c = np.zeros((B, H)), np.zeros((B, H))
        else:
            h, c = state
        outputs = []
        for t in range(T):
            # 1. 一次矩阵乘法算出 4 个门的预激活值
            gates = x[:, t] @ self.W_ih.T + self.b_ih + h @ self.W_hh.T + self.b_hh
            i, f, g, o = np.split(gates, 4, axis=1)
            i, f, g, o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
            # 2. 更新细胞状态: 遗忘旧信息 + 写入新信息
            c = f * c + i * g
            # 3. 输出门控制暴露多少细胞状态
            h = o * np.tanh(c)
            outputs.append(h)
        return np.stack(outputs, axis=1), (h, c)


class GRU(_RecurrentBase):
    num_gates = 3

    def forward(self, x, h0=None):
        B, T, _ = x.shape
        h = np.zeros((B, self.hidden_size)) if h0 is None else h0
        outputs = []
        for t in range(T):
            # x 部分与 h 部分要分开算，因为 r 只作用在 h 的 n 分支上
            gi = x[:, t] @ self.W_ih.T + self.b_ih
            gh = h @ self.W_hh.T + self.b_hh
            i_r, i_z, i_n = np.split(gi, 3, axis=1)
            h_r, h_z, h_n = np.split(gh, 3, axis=1)
            r = sigmoid(i_r + h_r)       # 重置门
            z = sigmoid(i_z + h_z)       # 更新门
            n = np.tanh(i_n + r * h_n)   # 候选隐状态
            h = (1 - z) * n + z * h
            outputs.append(h)
        return np.stack(outputs, axis=1), h


def _load_to_torch(ours, t_layer):
    import torch
    with torch.no_grad():
        t_layer.weight_ih_l0.copy_(torch.from_numpy(ours.W_ih))
        t_layer.weight_hh_l0.copy_(torch.from_numpy(ours.W_hh))
        t_layer.bias_ih_l0.copy_(torch.from_numpy(ours.b_ih))
        t_layer.bias_hh_l0.copy_(torch.from_numpy(ours.b_hh))


if __name__ == '__main__':
    import torch

    B, T, I, H = 2, 5, 3, 4
    x = np.random.default_rng(42).standard_normal((B, T, I))
    tx = torch.from_numpy(x)

    for ours, t_cls in [(RNN(I, H), torch.nn.RNN),
                        (LSTM(I, H), torch.nn.LSTM),
                        (GRU(I, H), torch.nn.GRU)]:
        t_layer = t_cls(I, H, batch_first=True).double()
        _load_to_torch(ours, t_layer)
        out, _ = ours.forward(x)
        t_out, _ = t_layer(tx)
        name = type(ours).__name__
        print(f"{name:<4} output shape: {out.shape}, match torch: {np.allclose(out, t_out.detach().numpy())}")
