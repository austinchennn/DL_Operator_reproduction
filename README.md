[English](README.md) | [中文](README_ZH.md)

# DL Operator Reproduction

Pure Python implementation of core deep learning operators

---

## Contents

### 1. Convolution
- [Convolution 2D](1.Convolution卷积操作.py) - Multi-channel 2D convolution with padding and stride support

### 2. Activation Functions
- [Softmax](2.Activation/2.1_Softmax实现.ipynb) - Numerically stable softmax implementation
- [Sigmoid](2.Activation/2.2_Sigmond实现.ipynb) - Sigmoid activation function
- [ReLU](2.Activation/2.3_ReLU实现.ipynb) - Rectified Linear Unit
- [SiLU](2.Activation/2.4_SiLU实现.ipynb) - Sigmoid Linear Unit (Swish)
- [GELU](2.Activation/2.5_GELU实现.ipynb) - Gaussian Error Linear Unit
- [SinLU](2.Activation/2.6_SinLU.ipynb) - Sinusoidal Linear Unit
- [SwiGLU](2.Activation/2.7_SwiGLU实现(Pytorch).ipynb) - Swish Gated Linear Unit (PyTorch)

### 3. Normalization
- [Layer Normalization](3.Normalization/3.1_Layer_Norm实现.py) - Layer-wise normalization with learnable parameters
- [RMS Normalization](3.Normalization/3.2_RMS_Norm实现.py) - Root Mean Square Layer Normalization

### 4. Attention Mechanisms
- [Grouped Query Attention](4.Attention/GroupedQueryAttention.py) - GQA: Memory-efficient attention with key/value head sharing

### 5. Layers
- [FFN (SwiGLU)](5.Layer/5.1_FFN/2.7_SwiGLU实现(Pytorch).ipynb) - Swish Gated Linear Unit feed-forward network
- [Embedding](5.Layer/5.2_Embedding实现.py) - Lookup-table embedding (forward + backward) and sinusoidal positional encoding
- [RNN / LSTM / GRU](5.Layer/5.3_RNN实现.py) - Recurrent layers with PyTorch-compatible weight layout
- [CNN](5.Layer/5.4_CNN实现.py) - im2col-based Conv2d (stride / padding / dilation), MaxPool2d, AvgPool2d

### 7. Optimizers
- [Optimizers Overview](7.Optimzer/README.md) - Gradient Descent, Momentum, Adagrad, RMSProp, Adam implementation and visualization.

---

## Papers

- [[3.1_Paper] Root Mean Square Layer Normalization](<papers/[3.1_Paper] Root Mean Square Layer Normalization.pdf>)

---

## Requirements

numpy>=1.24.0
torch>=2.1.0
jupyter>=1.0.0
matplotlib>=3.7.0
scipy>=1.11.0
