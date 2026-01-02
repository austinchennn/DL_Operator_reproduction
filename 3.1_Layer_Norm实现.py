import numpy as np
def layer_norm(x, gemma, beta, eps=1e-5):
    """
    x: 输入向量
    gamma: 可学习的缩放参数 (对应公式里的 γ)
    beta: 可学习的位移参数 (对应公式里的 β)
    """
    # 1. 计算均值 μ
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # 2. 计算方差 σ^2
    var = np.var(x, axis=-1, keepdims=True)
    
    # 3. 标准化: (x - μ) / sqrt(σ^2 + eps)
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    # 4. 线性变换: γ * x_hat + β
    return gamma * x_hat + beta
# 示例用法
x = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
gamma = np.array([1.0, 1.0, 1.0])
beta = np.array([0.0, 0.0, 0.0])
output = layer_norm(x, gamma, beta)

print("LayerNorm Output:", output)