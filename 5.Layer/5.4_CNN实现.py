# 5.4 卷积层 Conv2d + 池化层 MaxPool2d / AvgPool2d
# 与 1.Convolution卷积操作.py 的纯循环单输出通道版本不同，这里实现完整的批量、多输出通道卷积层，
# 并用 im2col 把卷积转成一次矩阵乘法 (工业框架常用的做法)。
#
# 输入 x: (N, C_in, H, W)，卷积核 weight: (C_out, C_in, K_h, K_w)，bias: (C_out,)
# H_out = (H + 2*padding - dilation*(K_h-1) - 1) // stride + 1   (W_out 同理)
#
# im2col:
#   1) 对输入补 0
#   2) 把每个输出位置对应的感受野 (C_in, K_h, K_w) 拉平成一列 -> cols: (N, C_in*K_h*K_w, H_out*W_out)
#   3) 卷积核拉平成 (C_out, C_in*K_h*K_w)，与 cols 做矩阵乘法即得输出
import numpy as np


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def im2col(x, kernel_size, stride=1, padding=0, dilation=1, pad_value=0.0):
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    N, C, H, W = x.shape
    H_out = (H + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    W_out = (W + 2 * pw - dw * (kw - 1) - 1) // sw + 1

    # 1. 补边
    x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=pad_value)

    # 2. 对卷积核内每个偏移 (i, j)，一次性取出所有输出位置对应的像素
    cols = np.empty((N, C, kh, kw, H_out, W_out), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            h0, w0 = i * dh, j * dw
            cols[:, :, i, j] = x_pad[:, :, h0:h0 + sh * H_out:sh, w0:w0 + sw * W_out:sw]
    return cols.reshape(N, C * kh * kw, H_out * W_out), (H_out, W_out)


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 bias=True, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.kernel_size = _pair(kernel_size)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        kh, kw = self.kernel_size
        # 与 PyTorch 一致: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        k = 1.0 / np.sqrt(in_channels * kh * kw)
        self.weight = rng.uniform(-k, k, (out_channels, in_channels, kh, kw))
        self.bias = rng.uniform(-k, k, out_channels) if bias else None

    def forward(self, x):
        N = x.shape[0]
        C_out = self.weight.shape[0]
        cols, (H_out, W_out) = im2col(x, self.kernel_size, self.stride, self.padding, self.dilation)
        # (C_out, C_in*K_h*K_w) @ (N, C_in*K_h*K_w, L) -> (N, C_out, L)
        out = self.weight.reshape(C_out, -1) @ cols
        if self.bias is not None:
            out += self.bias[None, :, None]
        return out.reshape(N, C_out, H_out, W_out)


class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = _pair(kernel_size)
        self.stride = self.kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x):
        N, C = x.shape[:2]
        kh, kw = self.kernel_size
        # 补 -inf 保证补边位置永远不会被选为最大值
        cols, (H_out, W_out) = im2col(x, self.kernel_size, self.stride, self.padding, pad_value=-np.inf)
        # 池化逐通道进行: (N, C, K_h*K_w, L) 上沿窗口维取最大
        return cols.reshape(N, C, kh * kw, -1).max(axis=2).reshape(N, C, H_out, W_out)


class AvgPool2d(MaxPool2d):
    def forward(self, x):
        N, C = x.shape[:2]
        kh, kw = self.kernel_size
        # 与 PyTorch 默认 count_include_pad=True 一致: 补的 0 也计入分母
        cols, (H_out, W_out) = im2col(x, self.kernel_size, self.stride, self.padding)
        return cols.reshape(N, C, kh * kw, -1).mean(axis=2).reshape(N, C, H_out, W_out)


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    x = np.random.default_rng(42).standard_normal((2, 3, 7, 7))
    tx = torch.from_numpy(x)

    for cfg in [dict(kernel_size=3, stride=1, padding=1),
                dict(kernel_size=3, stride=2, padding=0),
                dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), dilation=2)]:
        conv = Conv2d(3, 8, **cfg)
        out = conv.forward(x)
        t_out = F.conv2d(tx, torch.from_numpy(conv.weight), torch.from_numpy(conv.bias),
                         stride=conv.stride, padding=conv.padding, dilation=conv.dilation)
        print(f"Conv2d {cfg} -> {out.shape}, match torch: {np.allclose(out, t_out.numpy())}")

    pool_cfg = dict(kernel_size=3, stride=2, padding=1)
    out = MaxPool2d(**pool_cfg).forward(x)
    print(f"MaxPool2d {pool_cfg} -> {out.shape}, match torch: "
          f"{np.allclose(out, F.max_pool2d(tx, **pool_cfg).numpy())}")
    out = AvgPool2d(**pool_cfg).forward(x)
    print(f"AvgPool2d {pool_cfg} -> {out.shape}, match torch: "
          f"{np.allclose(out, F.avg_pool2d(tx, **pool_cfg).numpy())}")

    # 一个最小的 CNN 特征提取器: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> 全局平均池化
    conv1, conv2 = Conv2d(3, 8, 3, padding=1, seed=1), Conv2d(8, 16, 3, padding=1, seed=2)
    h = MaxPool2d(2).forward(np.maximum(conv1.forward(x), 0))
    h = np.maximum(conv2.forward(h), 0)
    feat = h.mean(axis=(2, 3))
    print("CNN feature shape:", feat.shape)
