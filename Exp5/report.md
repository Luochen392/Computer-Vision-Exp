# 实验五：CNN-卷积神经网络
202410315051 人工智能242 周落尘
## 一、实验目的
- 掌握卷积神经网络（CNN）基本结构，理解卷积、池化及全连接层的作用，使用PyTorch搭建LeNet-5模型。
- 熟悉CNN模型在MNIST数据集上的完整训练与测试流程，掌握交叉熵损失与Adam优化器的实际应用。
- 学会可视化卷积与池化特征图，直观理解特征提取机制，并通过多维度指标评估与对比模型性能。

## 二、实验内容
### 1.导入必要库及环境配置
- `torch`：PyTorch核心库，提供张量运算与GPU加速。
- `torch.nn as nn`：神经网络模块，提供卷积、全连接等基础层。
- `F (functional)`：函数式接口，提供激活、池化及损失函数。
- `optim`：优化器模块，提供 Adam 等算法更新参数。
- `datasets, transforms`：视觉工具包，用于下载数据集与格式转换。
- `matplotlib.pyplot`：绘图库，用于可视化训练曲线与特征图。
- `numpy as np`：科学计算库，辅助矩阵运算与数组处理。
- `confusion_matrix`：计算并生成混淆矩阵，用于评估模型。
> 其中在vscode环境下：
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

## 三、实验结果与分析

