import numpy as np

def rms_norm(x, gamma, eps=1e-8):
    """
    x: 输入向量
    gamma: 可学习的缩放参数 (对应公式里的 γ)
    """
    # 1. 计算平方的均值
    mean_square = np.mean(x ** 2, axis=-1, keepdims=True)
    # 2. 计算均方根 (RMS)
    rms = np.sqrt(mean_square + eps)
    # 3. 标准化: x / RMS
    x_hat= x / rms
    # 4. 线性变换: γ * x_hat
    return gamma * x_hat
# 示例用法
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
gamma = np.array([1.0, 1.0, 1.0])
output = rms_norm(x, gamma) 
print("RMSNorm Output:", output)