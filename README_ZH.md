[English](README.md) | [中文](README_ZH.md)

# DL 操作符复现

深度学习核心操作符的纯 Python 实现

---

## 目录

### 1. 卷积操作
- [卷积 2D](1.Convolution卷积操作.py) - 多通道2D卷积，支持padding和stride

### 2. 激活函数
- [Softmax](2.Activation/2.1_Softmax实现.ipynb) - 数值稳定的softmax实现
- [Sigmoid](2.Activation/2.2_Sigmond实现.ipynb) - Sigmoid激活函数
- [ReLU](2.Activation/2.3_ReLU实现.ipynb) - 修正线性激活函数
- [SiLU](2.Activation/2.4_SiLU实现.ipynb) - Sigmoid线性激活函数 (Swish)
- [GELU](2.Activation/2.5_GELU实现.ipynb) - 高斯误差线性激活函数
- [SinLU](2.Activation/2.6_SinLU.ipynb) - 正弦线性激活函数
- [SwiGLU](2.Activation/2.7_SwiGLU实现(Pytorch).ipynb) - Swish门控线性激活函数 (PyTorch)

### 3. 归一化
- [层归一化](3.Normalization/3.1_Layer_Norm实现.py) - 带可学习参数的层级归一化
- [RMS归一化](3.Normalization/3.2_RMS_Norm实现.py) - 均方根层归一化

### 4. 注意力机制
- [分组查询注意力](4.Attention/GroupedQueryAttention.py) - GQA: 通过键值头共享实现的内存高效注意力

### 5. 前馈网络
- [SwiGLU](5.FFN/2.7_SwiGLU实现(Pytorch).ipynb) - Swish门控线性激活函数实现

---

## 论文

- [[3.1_Paper] Root Mean Square Layer Normalization](<papers/[3.1_Paper] Root Mean Square Layer Normalization.pdf>)
