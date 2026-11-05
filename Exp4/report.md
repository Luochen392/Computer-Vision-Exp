<img width="381" height="329" alt="image" src="https://github.com/user-attachments/assets/c3bab57f-6d8d-43c9-994e-2ff1bd9e1ecd" /># 实验四 全连接神经网络
202410315051-人工智能242-周落尘
## 一、实验目的
- ### 掌握深度学习框架的使用：
  熟悉 PyTorch 框架的基本操作，包括张量（Tensor）的创建与变形、计算设备（CPU/GPU）的管理以及 torchvision 数据集加载工具的使用。
- ### 理解并构建全连接神经网络（MLP）：
  掌握多层感知机的网络结构设计，理解输入层、隐藏层、输出层的维度映射关系，以及 ReLU 激活函数在引入非线性特征中的作用。
- ### 掌握模型训练的标准范式：
  理解深度学习模型的“前向传播计算损失 → 反向传播计算梯度 → 优化器更新权重”的完整闭环，熟悉交叉熵损失函数（CrossEntropyLoss）和随机梯度下降（SGD）优化器的工作原理。
- ### 学会模型评估与性能分析
  掌握在测试集上评估模型泛化能力的方法，并能利用 Matplotlib 等可视化工具绘制 Loss 曲线、混淆矩阵（Confusion Matrix）及预测统计图，从定性和定量两个维度分析模型的分类效果与误判分布。

## 二、实验内容
### 1.导入必要库
- `torch`：PyTorch 的核心库。用于张量（Tensor）的创建与运算。
- `numpy`：科学计算基础库。用于处理多维数组矩阵运算。
- `torch.nn`：神经网络模块。包含了构建网络层所需的各类组件（如全连接层 Linear、卷积层 Conv2d）、损失函数（如 CrossEntropyLoss 等）。
- `torch.nn.functional`：函数式接口。里面包含了激活函数（F.relu）、池化操作，以及函数式的损失计算用。
- `torch.optim` '：优化器模块。包含了用于更新模型权重的各种优化算法，如随机梯度下降（SGD）、Adam 等。
- `matplotlib.pyplot`: 最常用的绘图库。在这里专门用于将训练过程中的 Loss 下降曲线、测试准确率，以及混淆矩阵等结果画成直观的图表，方便分析。
- `torchvision` PyTorch 针对视觉任务提供的工具包：
  - `datasets`：内置了常用的经典数据集（如实验中的 MNIST 手写数字数据集），提供了一键下载、读取和封装的功能。
  - `transforms`：数据预处理工具箱。用于对图像进行转换，比如把图片转成张量格式（ToTensor()）、裁剪、翻转以及标准化（Normalize）等操作。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
```

### 2.设置超参数和设备
> batch_size = 2048：设定每次送入模型的样本数量，以提升并行计算效率并稳定梯度更新。
> device = ...：自动检测并优先使用GPU加速计算，若无可用GPU则退回CPU。

```python
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3.加载数据
- 训练数据加载器 train_loader：加载 MNIST 训练集，按指定批次大小打乱并封装为可迭代对象
- 测试数据加载器 test_loader：加载 MNIST 测试集，同样按批次封装用于最终评估
- 数据预处理 transforms.ToTensor()：将图像转换为 PyTorch 张量，并将像素值归一化至 0~1 区间
- 数据信息验证 print()：输出数据集样本总数、Batch 划分数量及单批次张量维度，确保输入正确
```python
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                   ])),
    batch_size=batch_size, shuffle=True)

print("=== 数据集信息 ===")
# 1. 输出总样本数
print(f"训练集总样本数: {len(train_loader.dataset)}")
print(f"测试集总样本数: {len(test_loader.dataset)}")

# 2. 输出 Batch 数量
print(f"训练集 Batch 数量: {len(train_loader)}")
print(f"测试集 Batch 数量: {len(test_loader)}")

# 3. 抽取一个 Batch 查看数据维度形状
images, labels = next(iter(train_loader))
print("\n=== 数据维度 ===")
print(f"图像张量形状 (Batch, Channel, Height, Width): {images.shape}")
print(f"标签张量形状: {labels.shape}")
```

输出：

<div align="center">
  <img alt="output_image" src="https://github.com/user-attachments/assets/6229efe7-d88c-4255-98af-029431b5cf18" />
</div>

### 4.创建MLP模型

- 首先，需要先定义一个模型类 class mlp(nn.Module)，nn.Module：所有神经网络模块的基类，提供参数管理、GPU转移、序列化等功能
- 然后定义初始化方法 __init__，这里的代码中定义了两层的全连接层：
  - self.l1 = nn.Linear(784, 128): 输入维度: 784 (28×28) → 输出维度: 128 (隐藏层神经元数量)
  - self.l2 = nn.Linear(128, 10): 输入维度: 128 (隐藏层神经元数量) → 输出维度: 10 (对应10个数字类别0-9)
- 再定义前向传播 forward：
  - a1 = self.l1(x): 线性变换 x @ W1 + b1
  - x1 = F.relu(a1): ReLU激活函数 max(0, a1)
  - a2 = self.l2(x1): 线性变换 x1 @ W2 + b2
  - return a2: 输出层 (直接返回未经过 Softmax 的原始得分 Logits)
- 最后创建优化器并验证结构：
  - model = mlp().to(device): 先实例化MLP模型把模型参数移动到GPU/CPU设备上
  - optimizer = optim.SGD(model.parameters(), lr=0.01): 然后使用PyTorch内置优化器，采用随机梯度下降算法和0.01的学习率
  - print(model): 打印模型结构，直观验证网络层的维度与设计是否符合预期
 
```python
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2 = self.l2(x1)
        return a2

model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

print("基础 MLP 模型结构：")
print(model)
```

输出：

<div align="center">
  <img alt="mlp_image" src="https://github.com/user-attachments/assets/b51581db-e463-41ce-a39f-a040b02944d1" />
</div>

### 5.模型实例化与优化器设置 

- 实例化模型并加载设备 model = mlp().to(device)：创建多层感知机实例，并通过 .to() 将模型的所有参数（权重和偏置）移动至 GPU 或 CPU 硬件设备上，准备计算。
- 配置优化器 optimizer = optim.SGD(model.parameters(), lr=0.01)：调用随机梯度下降算法，绑定模型需要更新的参数，并将学习率（lr）设定为 0.01，用于后续指导模型权重的更新方向与步长。

```python
model = mlp().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

### 6.模型迭代训练

- epochs = 15 : 定义模型对整个训练集迭代训练的总轮数为 15 轮
- model.train() : 将模型切换为训练模式（启用内部特定层的特性）
- optimizer.zero_grad() : 每次计算前清空历史梯度，防止梯度在不同批次间错误累加
- loss.backward() : 根据计算出的误差值执行反向传播，自动求解各参数的梯度
- optimizer.step() : 优化器根据梯度方向更新模型参数，完成一步学习
- train_losses.append(avg_loss) : 将计算出的平均损失存入列表，为后续绘制曲线提供数据

```python
epochs = 15
train_losses = []  # 用于记录每个 epoch 的 loss

print("开始训练...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()
        
        # 累加当前 batch 的 loss
        epoch_loss += loss.item()
        
    # 计算当前 epoch 的平均 loss (这样画出来的曲线更平滑准确)
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # 3. 输出前 15 个 epoch 结果 
    # 每一个 epoch 输出一次，现在只会输出 15 行记录
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
```
> train_losses.append(avg_loss) : 将计算出的平均损失存入列表，为后续绘制曲线提供数据

输出：
<div align="center">
  <img alt="result_image" src="https://github.com/user-attachments/assets/230db31e-df8a-48eb-9860-6fa6a7b1a95b" />
</div>

Loss曲线图：

> 以下为loss曲线绘制代码

```python
%matplotlib inline
import matplotlib.pyplot as plt

# 绘制 Loss 曲线图
plt.figure(figsize=(8, 5))

# 绘制曲线：带圆点(o)，线宽2，使用深蓝色
plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='#1f77b4', linewidth=2)

# 设置标题和坐标轴标签
plt.title('Training Loss Curve (15 Epochs)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 开启网格线
plt.grid(True, linestyle='-', alpha=0.7)

# 显示图片
plt.tight_layout()
plt.show()
```

输出：
<div align="center">
  <img alt="loss_curve_image" src="https://github.com/user-attachments/assets/103f545d-701f-428a-b046-48dd1caa6fdc" />
</div>

### 7.测试模型
```python
model.eval()
correct = 0
test_loss = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

test_loss = test_loss / (batch_idx + 1)
acc = correct / len(test_loader.dataset)
print('测试结果 - loss:{:.4f}, acc:{:.4f}'.format(test_loss, acc))
```
输出：
<div align="center">
  <img width="475" height="34" alt="image" src="https://github.com/user-attachments/assets/f11740b0-e4a5-441c-a14c-9c9dde0dce43" />
</div>

## 三、实验结果与分析
### 1.原始模型结果
原始的二层感知机结构为 784 → 128 → 10，采用 ReLU 激活函数与 SGD 优化器。在 10 轮训练后，模型的最终测试准确率约为 90.52%。
从损失下降曲线可以看出，模型能够稳定收敛，但下降速度较慢，且在后期趋于平缓，说明模型的表达能力有限。
### 2.改进模型结果
改进后的模型结构为 784 → 256 → 128 → 64 → 10，并引入以下优化：
- 激活函数由 ReLU 改为 GELU，更平滑的梯度使模型收敛更稳定；
- 每层后加入 BatchNorm，缓解梯度消失并加快收敛；
- 引入 Dropout(0.3)，随机失活部分神经元以防止过拟合；
- 优化器由 SGD 改为 Adam，自适应学习率提高了训练效率。
经过训练后，模型在测试集上达到 97.95% 的准确率，损失值仅为 0.0679。从训练曲线看，改进模型在前几轮即可快速降低损失，收敛速度明显加快。混淆矩阵的对角线更为清晰，错误样本明显减少，表明模型的泛化性能显著提升。
### 3.结论：
通过网络结构优化与正则化手段，模型在分类准确率、收敛速度与稳定性上均有显著提升，验证了深层结构与合理正则化的重要性。

## 四、实验总结与心得
本实验通过从最基础的两层感知机入手，逐步改进为具有更强表达能力的多层感知机，使我对深度学习中模型结构、激活函数、优化方法和正则化策略有了更深入的理解。
在实验中，我发现：
- 网络深度与性能呈正相关，但需配合适当的归一化和正则化；
- 激活函数的选择 对模型性能影响显著，GELU 相较于 ReLU 在梯度平滑性上表现更优；
- BatchNorm 与 Dropout 的结合可以在提高模型稳定性的同时有效防止过拟合；
- Adam 优化器 收敛速度更快、对学习率敏感度更低，更适合深层网络。

总体而言，本实验让我系统掌握了多层感知机的原理与实现方法，并通过对比分析学会了如何从结构、优化与正则化三个角度改进模型性能。这不仅提升了我对神经网络的理解，也为后续更复杂的模型（如卷积神经网络、Transformer 等）奠定了坚实的基础。
