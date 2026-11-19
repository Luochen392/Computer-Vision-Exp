# 实验五：CNN-卷积神经网络

202410315051-人工智能242-周落尘

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

### 5. 模型训练与测试主循环

#### 训练阶段：
- 核心作用：让模型从数据中学习。
- 流程概述：将模型设为 train() 模式，遍历训练集批次。通过前向传播计算预测值，利用交叉熵求出误差（Loss），再执行反向传播（backward）计算梯度，最后使用 Adam 优化器（step）更新网络权重。整个过程中累加误差，计算并记录本轮的平均训练 Loss。

#### 测试阶段：
- 核心作用：评估模型在未见过的数据上的真实水平（不更新权重）。
- 流程概述：将模型切换为 eval() 模式并关闭梯度计算（torch.no_grad）以节省内存。遍历测试集进行前向推理，统计预测正确的样本数以计算准确率（Acc）和测试 Loss。同时，在最后一轮（Epoch 30）额外收集所有的预测标签和真实标签，为后续绘制混淆矩阵提供数据。

```python
epochs = 30
train_losses, test_losses, test_accs = [], [], []
all_preds, all_labels = [], [] # 用于最后绘制混淆矩阵

print("\n开始训练...")
for epoch in range(epochs):
    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)

    # --- 测试阶段 ---
    model.eval()
    correct = 0
    testloss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            testloss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            
            # 收集全部结果用于混淆矩阵
            if epoch == epochs - 1: # 只在最后一个 epoch 收集，节省内存
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

    acc = correct / len(testloader.dataset)
    test_loss = testloss / (batch_idx + 1)
    
    test_losses.append(test_loss)
    test_accs.append(acc)
    
    # 输出日志
    print('epoch:{}, train_loss:{:.4f}, test_loss:{:.4f}, acc:{:.4f}'.format(epoch + 1, train_loss, test_loss, acc))
```

输出：
```python
开始训练...

epoch:1, train_loss:2.3052, test_loss:2.3023, acc:0.1135
epoch:2, train_loss:2.2676, test_loss:1.9824, acc:0.2450
epoch:3, train_loss:1.1738, test_loss:0.5603, acc:0.8025
epoch:4, train_loss:0.3317, test_loss:0.2113, acc:0.9360
epoch:5, train_loss:0.1725, test_loss:0.1511, acc:0.9513
epoch:6, train_loss:0.1252, test_loss:0.1107, acc:0.9665
epoch:7, train_loss:0.0974, test_loss:0.0794, acc:0.9748
epoch:8, train_loss:0.0792, test_loss:0.0737, acc:0.9780
epoch:9, train_loss:0.0706, test_loss:0.0699, acc:0.9787
epoch:10, train_loss:0.0637, test_loss:0.0578, acc:0.9826
epoch:11, train_loss:0.0567, test_loss:0.0504, acc:0.9846
epoch:12, train_loss:0.0492, test_loss:0.0473, acc:0.9856
epoch:13, train_loss:0.0462, test_loss:0.0554, acc:0.9819
epoch:14, train_loss:0.0416, test_loss:0.0467, acc:0.9855
epoch:15, train_loss:0.0426, test_loss:0.0506, acc:0.9837
epoch:16, train_loss:0.0405, test_loss:0.0495, acc:0.9860
epoch:17, train_loss:0.0350, test_loss:0.0426, acc:0.9865
epoch:18, train_loss:0.0332, test_loss:0.0499, acc:0.9847
epoch:19, train_loss:0.0325, test_loss:0.0444, acc:0.9865
epoch:20, train_loss:0.0308, test_loss:0.0362, acc:0.9885
epoch:21, train_loss:0.0286, test_loss:0.0405, acc:0.9883
epoch:22, train_loss:0.0279, test_loss:0.0448, acc:0.9879
epoch:23, train_loss:0.0255, test_loss:0.0491, acc:0.9852
epoch:24, train_loss:0.0249, test_loss:0.0446, acc:0.9876
epoch:25, train_loss:0.0242, test_loss:0.0446, acc:0.9865
epoch:26, train_loss:0.0235, test_loss:0.0385, acc:0.9885
epoch:27, train_loss:0.0193, test_loss:0.0496, acc:0.9859
epoch:28, train_loss:0.0189, test_loss:0.0480, acc:0.9860
epoch:29, train_loss:0.0186, test_loss:0.0421, acc:0.9874
epoch:30, train_loss:0.0175, test_loss:0.0400, acc:0.9884
```

### 6. 绘制 Loss 和 Acc 曲线图

模型训练过程的数据可视化，展示模型的学习效果：
- （Loss 曲线）：绘制了 Train Loss（蓝线）与 Test Loss（红线）随 Epoch 的变化。可以直观对比训练集和测试集的误差下降趋势。
- （Accuracy 曲线）：绘制了 Test Accuracy（绿线）随 Epoch 的变化，展示模型在测试集上准确率的提升过程。

核心作用：通过曲线的走势，帮助开发者直观判断模型是否收敛、学习率是否合适，以及是否存在过拟合现象（例如两条 Loss 曲线是否出现较大分离）。

```python
plt.figure(figsize=(14, 5))

# 左图：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue', marker='o', markersize=4)
plt.plot(range(1, epochs+1), test_losses, label='Test Loss', color='red', marker='s', markersize=4)
plt.title('Loss Curve (Train vs Test)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 右图：Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), test_accs, label='Test Accuracy', color='green', marker='^')
plt.title('Test Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

```

输出：
<div align="center">
  <img width="1400" height="500" alt="curve" src="https://github.com/user-attachments/assets/f406bdeb-8803-4db3-87c8-d2a561912ebd" />
</div>

### 7. 绘制混淆矩阵

模型的混淆矩阵可视化，用以直观分析模型在各个类别上的分类表现和误判情况。具体概括如下：
- 数据计算：调用 confusion_matrix，利用收集到的真实标签（all_labels）和预测标签（all_preds）生成 10×10的混淆矩阵。
- 热力图渲染：使用 plt.imshow 将矩阵渲染为蓝色调的热力图（cmap='Blues'），并添加颜色条（colorbar）辅助查看数值大小。
- 数值标注：通过双层循环遍历矩阵的每一个格子，将具体数字填在中心。巧妙利用 thresh 阈值判断背景深浅：深色背景填白字，浅色背景填黑字，确保数字清晰可见。
- 坐标轴设置：设定 X 轴为“预测标签”，Y 轴为“真实标签”，刻度标尺设为 0-9。

核心作用：在这张图中，对角线上的深色方块代表预测正确的数量；而非对角线上的浅色方块则暴露了模型的弱点（例如，如果第 4 行第 9 列有数字，说明模型容易把真实的 4 错认成 9）。

```python
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('LeNet Confusion Matrix')
plt.colorbar()

# 在矩阵格子里填上具体数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(range(10))
plt.yticks(range(10))
plt.tight_layout()
plt.show()
```

输出：
<div align="center">
  <img width="800" height="600" alt="confusion matrix" src="https://github.com/user-attachments/assets/d319d9b2-ac60-4257-8cab-08ca328ee28e" />
</div>

### 8.卷积与池化特征图可视化

神经网络内部的过程可视化，直观展示卷积层和池化层是如何一步步提取图像特征的。简要概括如下：
- 数据提取：从测试集中抽取 5 张手写数字图片（x_vis）。
- 中间层前向传播：手动执行两次 “卷积 → Sigmoid → 池化” 操作，单独截取并保存这两个阶段产生的特征图（feature1_pool 和 feature2_pool）。
- 张量转换：将模型中的张量（Tensor）脱离计算图（detach），移至 CPU 并转换为 NumPy 数组，以便用 Matplotlib 绘图。
- 网格绘图：创建一个 3×5 的画布：
    - 第一行（Original）：显示原始输入图像。
    - 第二行（Conv1+Pool）：显示第一层提取的特征。
    - 第三行（Conv2+Pool）：显示第二层提取的高维抽象特征。

> 可视化技巧：绘图时使用了 img[i].sum(0)，这是通过将多通道特征图在通道维度上求和，将其压缩成单通道矩阵，从而能用灰度图直观展现出来。


```python
# 重新取一个 batch 的测试数据用于可视化
dataiter = iter(testloader)
images, labels = next(dataiter)
x_vis = images[:5].to(device) # 取前5张

# 第一层：卷积 -> Sigmoid -> 池化
feature1 = torch.sigmoid(model.conv1(x_vis))
feature1_pool = F.avg_pool2d(feature1, kernel_size=2, stride=2)

# 第二层：卷积 -> Sigmoid -> 池化
feature2 = torch.sigmoid(model.conv2(feature1_pool))
feature2_pool = F.avg_pool2d(feature2, kernel_size=2, stride=2)

n = 5 
img = x_vis.detach().cpu().numpy()
f1 = feature1_pool.detach().cpu().numpy()
f2 = feature2_pool.detach().cpu().numpy()

fig, ax = plt.subplots(3, n, figsize=(15, 8))
for i in range(n):
    ax[0,i].imshow(img[i].sum(0), cmap='gray')
    ax[0,i].set_title("Original")
    ax[0,i].axis('off')
    
    ax[1,i].imshow(f1[i].sum(0), cmap='gray')
    ax[1,i].set_title("Conv1 + Pool")
    ax[1,i].axis('off')
    
    ax[2,i].imshow(f2[i].sum(0), cmap='gray')
    ax[2,i].set_title("Conv2 + Pool")
    ax[2,i].axis('off')

plt.tight_layout()
plt.show()
```

输出：
<div align="center">
  <img width="1500" height="800" alt="comparation" src="https://github.com/user-attachments/assets/c6b45d19-ed11-420e-b700-63abd349f0c8" />
</div>

## 三、实验结果与分析

