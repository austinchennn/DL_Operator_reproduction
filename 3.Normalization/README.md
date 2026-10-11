# Normalization / 归一化

## Contents / 目录

- [LayerNorm](3.1_Layer_Norm实现.py) - 层归一化
- [RMSNorm](3.2_RMS_Norm实现.py) - 均方根归一化
- [BatchNorm](3.3_Batch_Norm实现.py) - 批归一化（含 running 统计量与 train/eval 切换）

## Notes / 笔记

### BatchNorm vs LayerNorm

两者公式完全一样：

$$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

唯一的区别是：**均值 μ 和方差 σ² 沿哪个维度算。**

一句话概括：**BatchNorm 是“横向比较”（和同一批的其他样本比），LayerNorm 是“自己和自己比”（只看单个样本内部）。**

#### 1. Batch Normalization (BatchNorm)

- **计算方式：** 跨样本（Batch）归一化。在一个 Mini-Batch 中，针对同一个特征维度（Channel/Feature），计算所有样本的均值 μ 和方差 σ²。
- **直观理解：** 假设班里有 30 个学生（Batch Size = 30），每个人考了语文、数学、英语三科。BatchNorm 就是分别算全班 30 个人的语文平均分、数学平均分、英语平均分，然后**按科目**标准化。
- **适用场景：** 主要用于计算机视觉（CV）和卷积神经网络（CNN）。
- **核心缺陷：**
  - **严重依赖 Batch Size：** Batch Size 太小（如 2 或 4）时，算出的均值和方差不能代表全局数据分布，模型性能会明显下降。
  - **不适合变长序列：** NLP 里句子长度往往不同。硬用 BatchNorm 的话，不同位置的统计量很不稳定，而且长句会带来大量无意义的填充（Padding）数据干扰计算。

#### 2. Layer Normalization (LayerNorm)

- **计算方式：** 跨特征（Feature/Channel）归一化。对每个单独的样本，计算它所有特征维度的均值 μ 和方差 σ²。
- **直观理解：** 还是上面的例子，LayerNorm 是算**某一个学生自己**语文、数学、英语三科的平均分，然后只对这个学生自己的成绩做标准化，和其他 29 个学生无关。
- **适用场景：** 主要用于自然语言处理（NLP）、Transformer 架构（如大语言模型）和循环神经网络（RNN）。
- **核心优势：**
  - **完全不依赖 Batch Size：** 就算 Batch Size 为 1 也能正常计算，很适合显存受限时的小 Batch 训练。
  - **天然适配变长序列：** 它对单个时间步（Token）做归一化，序列长短不影响计算逻辑。

#### 3. 维度对比

**以 NLP/Transformer 为例**，输入形状为 `[N, L, D]`（N = Batch Size，L = 序列长度，D = 隐藏层维度）：

- BatchNorm 沿 `(N, L)` 求均值和方差，得到 **D 组**统计量。
- LayerNorm 沿 `(D)` 求均值和方差，得到 **N×L 组**统计量（每个样本的每个词都有自己独立的一组）。

**以 CV/CNN 为例**，输入形状为 `[N, C, H, W]`：

- BatchNorm 沿 `(N, H, W)` 求统计量，得到 **C 组**（每个通道一组）。
- LayerNorm 一般沿 `(C, H, W)` 求统计量，得到 **N 组**（每张图一组）。

| 输入形状 | BatchNorm 归一化维度 | 统计量个数 | LayerNorm 归一化维度 | 统计量个数 |
| --- | --- | --- | --- | --- |
| `[N, D]` | `N` | D | `D` | N |
| `[N, L, D]` | `N, L` | D | `D` | N×L |
| `[N, C, H, W]` | `N, H, W` | C | `C, H, W` | N |

#### 4. 全面对比

| 特性 | BatchNorm | LayerNorm |
| --- | --- | --- |
| **归一化方向** | 跨样本，同一特征 | 单样本内部，跨特征 |
| **依赖 Batch Size** | 是，小 Batch 效果差 | 否，Batch Size = 1 也可以 |
| **训练 / 推理行为** | **不同**：训练用当前 batch 统计量，推理用 `running_mean` / `running_var` | **相同**：两种模式都实时计算 |
| **额外状态** | 需要维护 `running_mean`、`running_var`（不参与梯度） | 无，只有 γ、β |
| **γ、β 形状** | `[C]`（每个通道一组） | `[D]`（归一化的特征维度） |
| **变长序列 / Padding** | 不友好，Padding 会污染统计量 | 天然支持 |
| **样本间耦合** | 有，一个样本的输出受同 batch 其他样本影响 | 无，样本之间完全独立 |
| **分布式训练** | 多卡时每卡 batch 变小，常需 SyncBN 同步统计量 | 无需同步 |
| **正则化效果** | 有轻微正则化效果（batch 统计量带来噪声） | 基本没有 |
| **典型场景** | CNN：ResNet、VGG 等图像模型 | Transformer、RNN、LLM |

#### 5. 场景区别：实战中怎么选？

- **用 BatchNorm：** 图像任务 + CNN + Batch Size 足够大（通常 ≥ 16/32）。卷积特征图同一个通道的不同位置语义相近，跨样本统计很稳定。另外推理时 BN 可以和前面的卷积层**融合**（把 γ、β、μ、σ 合进卷积权重），零额外开销。
- **用 LayerNorm：** 序列任务、Transformer、RNN、小 Batch 训练、在线推理（一次只来一个样本）。
- **RNN 为什么不用 BN：** 每个时间步的统计量不一样，要按时间步分别存 running 统计量，测试序列比训练序列长时就没有对应的统计量可用。

#### 6. 实现上的注意点（BatchNorm）

- **train / eval 必须切换：** 推理时忘了 `model.eval()`，BN 会继续用当前 batch 的统计量并更新 running 统计量，结果会不稳定。
- **running 统计量的更新：** `running = (1 - momentum) * running + momentum * batch_stat`，PyTorch 默认 `momentum = 0.1`（注意和优化器的 momentum 定义相反）。
- **方差的有偏/无偏：** 归一化当前 batch 用**有偏**方差（除以 m），更新 `running_var` 用**无偏**方差（除以 m - 1），与 PyTorch 一致。
- **Batch Size = 1 时：** 训练模式下 `[1, C]` 的输入方差为 0，BN 无法正常工作（PyTorch 会直接报错）。

#### 7. 延伸：RMSNorm

RMSNorm 可以看作**去掉均值中心化的 LayerNorm**：只除以均方根 $\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$，不减均值，也通常不要 β。计算更省，效果和 LN 差不多，所以 LLaMA 等大模型基本都用 RMSNorm。

#### 8. 其他变体（了解）

- **InstanceNorm：** 每个样本的每个通道单独归一化（沿 `H, W`），常用于风格迁移。
- **GroupNorm：** 把通道分成若干组，每组内沿 `(C/G, H, W)` 归一化，不依赖 Batch Size，适合检测、分割等小 Batch 的 CV 任务。G = 1 时等价于 LayerNorm，G = C 时等价于 InstanceNorm。
