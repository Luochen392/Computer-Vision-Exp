# 实验五：CNN-卷积神经网络
202410315051 人工智能242 周落尘
## 一、实验目的
- 掌握卷积神经网络（CNN）基本结构，理解卷积、池化及全连接层的作用，使用PyTorch搭建LeNet-5模型。
- 熟悉CNN模型在MNIST数据集上的完整训练与测试流程，掌握交叉熵损失与Adam优化器的实际应用。
- 学会可视化卷积与池化特征图，直观理解特征提取机制，并通过多维度指标评估与对比模型性能。

## 二、实验内容
### 1. 导入必要库及环境配置
- `torch`：PyTorch核心库，提供张量运算与GPU加速。
- `torch.nn as nn`：神经网络模块，提供卷积、全连接等基础层。
- `F (functional)`：函数式接口，提供激活、池化及损失函数。
- `optim`：优化器模块，提供 Adam 等算法更新参数。
- `datasets, transforms`：视觉工具包，用于下载数据集与格式转换。
- `matplotlib.pyplot`：绘图库，用于可视化训练曲线与特征图。
- `numpy as np`：科学计算库，辅助矩阵运算与数组处理。
- `confusion_matrix`：计算并生成混淆矩阵，用于评估模型。
> 在vscode环境下：
> - os：设置环境变量，防止 Windows 下 OpenMP 库冲突。
> - plt.rcParams：配置绘图参数，解决图表中文乱码及负号问题。

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
```
### 2. 超参数与数据加载
- `batch_size = 512`：设定每次送入模型训练的样本数量为 512 张，以平衡计算效率与内存占用。
- `device = ...`：自动检测并优先使用 GPU（CUDA）加速计算，若没有则退回 CPU，并打印当前使用的设备。
- `trainloader = ...`：自动下载 MNIST 训练集，将图像转为张量，并打乱顺序按批次封装，用于训练模型。
- `testloader = ...`：自动下载 MNIST 测试集，同样转为张量封装，用于后续评估模型的泛化能力。

```python
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")

trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)
```
输出：
```
当前使用的设备: cpu
```

### 3. 定义 LeNet-5 网络模型
> 本实验基于 PyTorch 构建了经典的 LeNet-5 架构，该网络通过局部感受野和权值共享机制，有效提取了图像的二维空间特征。整体网络可划分为 “特征提取器”（卷积与池化）与 “分类器”（全连接层）两大模块，其详细结构推演如下：
- 输入层：输入形状为 (Batch, 1, 28, 28) 的单通道灰度图。
- 第一层卷积 (conv1)：
  - 卷积核：5×5，padding=2
  - 输出尺寸：(28 + 2×2 - 5 + 1) = 28×28
  - 通道数：6
- 第一层池化：
  - 池化核：2×2，stride=2
  - 输出尺寸：28/2 = 14×14
  - 通道数：6
- 第二层卷积 (conv2)：
  - 卷积核：5×5
  - 输出尺寸：(14 - 5 + 1) = 10×10
  - 通道数：16
- 第二层池化：
  - 池化核：2×2，stride=2
  - 输出尺寸：10/2 = 5×5
  - 通道数：16
- 第一层全连接 (fc1)：
  - 输入：16×5×5
  - 输出：120
- 第二层全连接 (fc2)：
  - 输入：120
  - 输出：84
- 第三层全连接 (clf)：(MNIST只有10类，第三层作为分类层)
  - 输入：84
  - 输出：10

所以整体结构包括：
- 两层卷积和池化
  - 每层卷积后都接 Sigmoid 激活函数与平均池化
- 三层全连接层
  - FC1：将卷积特征映射到更高层次的语义空间
  - FC2：进一步抽象特征
  - FC3（分类层）：输出10类数字的预测得分

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入通道1, 输出通道6, 卷积核5x5, padding=2
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 输入通道6, 输出通道16, 卷积核5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 全连接层, 输入5*5*16=400, 输出120
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        # 全连接层, 输入120, 输出84
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        # 分类层, 输入84, 输出10
        self.clf = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # Layer 1: Conv1 -> Sigmoid -> AvgPool
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # Layer 2: Conv2 -> Sigmoid -> AvgPool
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # Layer 3: FC1 -> Sigmoid
        x = self.fc1(x)
        x = torch.sigmoid(x)

        # Layer 4: FC2 -> Sigmoid
        x = self.fc2(x)
        x = torch.sigmoid(x)

        # Layer 5: Output
        x = self.clf(x)
        return x
```

### 4. 实例化模型与优化器
- `model = Net().to(device)`：实例化刚刚定义好的 LeNet-5 模型，并将其所有参数加载到指定的硬件设备（优先 GPU，否则 CPU）上，准备计算。
- `optimizer = optim.Adam(...)`：配置 Adam 优化器，接管模型的所有参数（model.parameters()），并将学习率（lr）设定为 1e-2（即 0.01），用于后续在反向传播中自动更新模型权重。

```python
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
```
## 三、实验结果与分析

