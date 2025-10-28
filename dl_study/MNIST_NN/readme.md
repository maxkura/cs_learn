# MNIST 手写数字识别：自定义两层神经网络

本项目实现了一个**完全由 NumPy/Tensor 操作构建的两层全连接神经网络**，用于在 MNIST 手写数字数据集上进行图像分类。该项目不依赖 PyTorch 的高级模块（如 `nn.Module`, `optim`, `autograd`），而是手动实现了前向传播、损失计算、反向传播和参数更新的核心逻辑，非常适合学习深度学习基础原理。

## 🔍 项目概述

- **功能**: 训练一个简单的神经网络来识别手写数字 (0-9)。
- **框架**: 主要使用 `PyTorch` 进行张量运算和数据加载，但所有模型计算均为手动实现。
- **核心亮点**:
  - 手动实现 Affine 层、ReLU/Sigmoid 激活函数、Softmax + CrossEntropy Loss。
  - 实现了完整的反向传播算法及梯度下降优化过程。
  - 包含 He 初始化策略以加速收敛。
  - 数据预处理与可视化支持。

---

## 🧱 网络架构说明

| 层名       | 输入维度   | 输出维度    | 激活函数 |
|------------|-------------|--------------|-----------|
| Input      | 28×28=784   |              |           |
| Hidden     | 784         | 1024         | ReLU      |
| Output     | 1024        | 10           | Softmax   |

> 💡 注：虽然代码中包含 Sigmoid 类，但在当前网络结构中并未使用；实际使用的激活是 ReLU 和 Softmax。

---

## 📦 文件组成 & 核心类解析

### ✅ 核心 Python 脚本 (`main.py`)
此脚本完成了从数据准备到训练再到可视化的全流程操作：

#### 自定义组件：
1. **Affine Layer (`affine`)**  
   实现线性变换 $ y = xW + b $，并能返回对输入、权重、偏置的梯度。

2. **Activation Functions**
   - `ReLU`: max(0, x)，带有 backward 方法。
   - `Sigmoid`: σ(x) = 1/(1+exp(-x))，带 backward 方法。（未启用）

3. **Loss Function (`softmaxwithloss`)**
   结合 softmax 归一化与交叉熵损失，并提供 backward 接口输出初始误差项。

4. **Network Model (`twowisenetwork`)**
   - 构造双层感知机结构；
   - 支持正向推理（`.forward()`）、损失计算（`.loss()`）、误差反传（`.backward()`）和参数更新（`.update()`）；
   - 内部封装各子层对象以便统一管理其状态。

5. **Data Loader Setup**
   利用 `torchvision.datasets.MNIST` 下载并标准化 MNIST 数据集，并通过 DataLoader 分批次读入内存。

6. **Training Loop**
   循环执行 t=200 次迭代，在每次循环内完成一次 mini-batch 前向 → loss → backward → update 流程。

7. **Visualization Tools**
   提供展示样本图像的功能（matplotlib 绘图保存为 png 文件）。


