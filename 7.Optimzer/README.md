# Optimizers Implementation / 优化器实现

This module contains the implementation and reproduction of classic deep learning optimization algorithms, including visualization of their convergence paths on 2D contours and 3D loss surfaces.

本模块包含深度学习经典优化算法的实现与复现，并提供了它们在 2D 等高线和 3D 损失曲面上的收敛路径可视化。

## Contents / 目录

### 1. Gradient Descent (梯度下降)
- **Notebook**: [1_gradient_descent.ipynb](1_gradient_descent.ipynb)
- Basic implementation of Gradient Descent.
- 梯度下降算法的基础实现。

### 2. Momentum (动量法)
- **Notebook**: [2_Momentum.ipynb](2_Momentum.ipynb)
- Accelerates SGD in the relevant direction and dampens oscillations.
- 在相关方向加速 SGD 并抑制震荡。

![Momentum](graph/梯度下降对比-%20普通(红)%20vs%20动量法(蓝).png)

### 3. Adagrad
- **Notebook**: [3_Adagrad.ipynb](3_Adagrad.ipynb)
- Adaptive learning rate optimization. Adapts the learning rate to the parameters.
- 自适应学习率优化算法。根据参数调整学习率。

![Adagrad](graph/3D%20View-%20Adagrad能够更好地适应不同方向的梯度.png)

### 4. RMSProp
- **Notebook**: [4_RMSProp.ipynb](4_RMSProp.ipynb)
- Root Mean Square Propagation. Resolves Adagrad's radically diminishing learning rates.
- 均方根传播。解决了 Adagrad 学习率急剧下降的问题。

![RMSProp](graph/3D%20View-%20RMSProp%20能够有效地调整学习步伐.png)

### 5. Adam
- **Notebook**: [5.Adam.ipynb](5.Adam.ipynb)
- Adaptive Moment Estimation. Combines the advantages of Momentum and RMSProp.
- 自适应矩估计。结合了 Momentum 和 RMSProp 的优势。

![Adam](graph/3D%20View-%20Adam%20结合了%20Momentum%20和%20RMSProp%20的优势.png)

### 6. Optimizer Comparison (优化器大比拼)
- **Notebook**: [6_optimizer_comparison.ipynb](6_optimizer_comparison.ipynb)
- Comprehensive comparison of all optimizers on the challenging Rosenbrock function.
- 在具有挑战性的 Rosenbrock 函数上对所有优化器进行综合对比。

![Comparison](graph/Optimer%20Comparison.png)
