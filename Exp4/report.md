# 实验四 全连接神经网络
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
- batch_size = 2048：设定每次送入模型的样本数量，以提升并行计算效率并稳定梯度更新。
- device = ...：自动检测并优先使用GPU加速计算，若无可用GPU则退回CPU。

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

### 1.数据加载与训练过程分析

- 本次实验采用了 batch_size = 2048 的大批次加载方式，将 60000 张训练集划分为了 30 个 Batch，不仅充分利用了矩阵并行计算能力，也保证了梯度下降方向的稳定性。
- 从 15 个 Epoch 的训练日志与生成的 Loss 曲线可以看出：训练初期的 Loss 为 2.2720，随着迭代进行，Loss 值稳步且平滑地下降，在第 15 轮时降至 1.1029。整条曲线无任何剧烈震荡或反弹，这说明设定的学习率（0.01）与 SGD 优化器非常契合当前网络，模型正在健康、稳定地收敛，成功学习到了手写数字的底层特征。

### 2.测试结果与模型性能评估
将训练好的模型切换到评估模式（model.eval()）并在测试集上进行验证后，模型取得了较理想的准确率（约 90% 左右）。

结合可视化的混淆矩阵与预测统计图分析：
- 整体表现良好：大部分测试样本集中在混淆矩阵的深蓝色主对角线上，整体预测正确（Correct）的样本数远超错误（Incorrect）样本数，证明基础 MLP 模型具备较强的泛化能力。
- 局部误判存在：在分类统计图中可以看出，模型对数字 0、1 的识别率极高，但在数字 4 和 9 等类别上存在少量误判。这是因为基础的单隐藏层网络对复杂边缘特征的提取能力有限，且部分人的手写习惯会导致这两个数字特征相近。
### 3.结论：
通过本次基础实验，验证了单隐藏层全连接神经网络（784 → 128 → 10）在 MNIST 数据集上的有效性。虽然模型结构简单，但通过标准的训练范式依然能达到约 90% 的准确率，为后续的模型改进奠定了可靠的基础。

## 四、实验总结与心得
本次实验通过使用 PyTorch 从零搭建并训练了一个基础的多层感知机（MLP），让我对深度学习的完整工作流有了深刻理解，不仅停留在理论层面，更积累了宝贵的实操经验：
- 加深了对 PyTorch 数据流的理解：熟悉了 Dataset 与 DataLoader 的封装机制，深刻体会到了将数据展平（view）并转移到计算设备（.to(device)）是进入全连接层的必经之路。
- 掌握了神经网络的标准训练闭环：通过亲手编写训练循环，我真正理解了“前向传播求损失 → 清空梯度 → 反向传播求导 → 优化器更新”的底层逻辑。并学会了在 Jupyter 环境中通过打印维度信息（print(images.shape)）来排查维度不匹配的报错。
- 认识到了可视化分析的重要性：相比于只看一个最终的准确率数字，通过绘制 Loss 曲线可以判断是否过拟合；通过混淆矩阵可以精准定位模型“在哪些数字上容易搞错”，这为后续优化模型提供了明确的指导方向。
基于PyTorch构建MLP完成MNIST识别，利用Loss曲线与混淆矩阵分析性能，深入掌握了深度学习核心全流程。
