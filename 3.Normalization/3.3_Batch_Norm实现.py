import numpy as np
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        """
        num_features: 特征/通道数 C
        eps: 防止除零的小常数
        momentum: running 统计量的更新系数 (与 PyTorch 定义一致)
        """
        self.eps = eps
        self.momentum = momentum
        # 可学习参数 γ, β
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        # 推理时使用的全局统计量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def __call__(self, x):
        """
        x: (N, C) 或 (N, C, H, W)，统计量在除 C 以外的所有维度上计算
        """
        axes = (0,) if x.ndim == 2 else (0, 2, 3)
        shape = (1, -1) if x.ndim == 2 else (1, -1, 1, 1)

        if self.training:
            # 1. 用当前 batch 计算均值 μ 和方差 σ^2 (有偏估计)
            mean = np.mean(x, axis=axes)
            var = np.var(x, axis=axes)

            # 2. 更新 running 统计量 (running_var 用无偏估计)
            m = x.size // x.shape[1]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * m / (m - 1)
        else:
            # 推理时使用训练积累的全局统计量
            mean, var = self.running_mean, self.running_var

        # 3. 标准化: (x - μ) / sqrt(σ^2 + eps)
        x_hat = (x - mean.reshape(shape)) / np.sqrt(var.reshape(shape) + self.eps)

        # 4. 线性变换: γ * x_hat + β
        return self.gamma.reshape(shape) * x_hat + self.beta.reshape(shape)
# 示例用法
np.random.seed(0)
x = np.random.randn(4, 3, 5, 5)
bn = BatchNorm(num_features=3)
output = bn(x)
print("BatchNorm Output 每个通道的均值:", output.mean(axis=(0, 2, 3)))
print("BatchNorm Output 每个通道的方差:", output.var(axis=(0, 2, 3)))

bn.training = False
print("Eval Output shape:", bn(x).shape)

# 与 PyTorch 对比
try:
    import torch
    torch_bn = torch.nn.BatchNorm2d(3)
    torch_out = torch_bn(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    print("与 nn.BatchNorm2d 输出一致:", np.allclose(output, torch_out, atol=1e-5))
    print("running_var 一致:", np.allclose(bn.running_var, torch_bn.running_var.numpy(), atol=1e-5))
except ImportError:
    pass
