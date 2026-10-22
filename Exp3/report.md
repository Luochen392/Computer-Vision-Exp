# 实验三：图像关键点检测及图像拼接
202410315051-人工智能242-周落尘
## 一、实验目的
基于 SIFT 特征检测算法和 FLANN 特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接。掌握图像配准（单应性矩阵计算）和透视变换融合的基本原理与实现方法。
## 二、实验内容
### 1.导入必要库
- cv2 (OpenCV库)：用于图像读取、SIFT特征提取、FLANN匹配、单应性矩阵计算及透视变换等核心操作。
- numpy (Numpy库)：用于构建匹配点坐标矩阵、计算图像角点以及构建平移矩阵。
- matplotlib.pyplot：用于替换 cv2.imshow 以避免 Jupyter 内核崩溃，并实现高自由度的多子图对比排版。
- skimage.metrics：用于引入 PSNR 和 SSIM 算法，客观评估拼接结果与原图的相似度。

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
```

### 2. 图像预处理与特征提取
- 读取待拼接的原图 a.jpg（彩色基准图）与 b.jpg（待变换图）。
- 为了提升特征检测算法的运算效率，并消除色彩冗余对特征提取的干扰，将两张彩色图像转换为单通道灰度图。
- 随后，利用 matplotlib 绘制 2x2 的对比图阵，直观展示彩色原图与灰度图的差异。

```python
# 读取图像
img_a = cv2.imread("a.jpg")
img_b = cv2.imread("b.jpg")

# 转换格式 (注意这里必须定义 img_a_rgb 和 img_b_rgb)
img_a_rgb = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)
img_b_rgb = cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB)

# 转换为灰度图
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# 创建 2x2 画布
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 绘制四个子图
# 左上：Image A (Color)
axes[0, 0].imshow(img_a_rgb)  # 这里用到 img_a_rgb
axes[0, 0].set_title('Image A (Color)')
axes[0, 0].axis('off')

# 右上：Image B (Color)
axes[0, 1].imshow(img_b_rgb)  # 这里用到 img_b_rgb
axes[0, 1].set_title('Image B (Color)')
axes[0, 1].axis('off')

# 左下：Image A (Grayscale)
axes[1, 0].imshow(gray_a, cmap='gray')
axes[1, 0].set_title('Image A (Grayscale)')
axes[1, 0].axis('off')

# 右下：Image B (Grayscale)
axes[1, 1].imshow(gray_b, cmap='gray')
axes[1, 1].set_title('Image B (Grayscale)')
axes[1, 1].axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()
```
输出：
<div align="center">
  <img alt="original_images" src="https://github.com/user-attachments/assets/1d650909-3660-4d61-a3bb-476a170fdebe" />
  <img alt="gray_images" src="https://github.com/user-attachments/assets/93d1ea67-80eb-47a0-b8a6-ffd7e93bfdef" />
</div>

### 3. 特征匹配与筛选
- 利用 SIFT 算法 提取两幅图像的关键点（Keypoints）与高维特征描述符（Descriptors）。SIFT 特征对旋转、尺度缩放和亮度变化具有极强的鲁棒性。
- 进入特征匹配与预筛选阶段。算法构建了 FLANN (Fast Library for Approximate Nearest Neighbors) 匹配器进行快速近似最近邻搜索。
- 具体步骤与原理如下：
  1. KNN 粗匹配：使用 knnMatch 函数，为图 A 中的每个特征点在图 B 中寻找 k=2（即 2 个）距离最近的特征点，得到初始匹配集。
  2. Lowe's 比率测试（Ratio Test）：这是一种经典的剔除误匹配策略。对于每一个特征点，系统比较其“最近邻距离”和“次近邻距离”。如果两者差距不够大（即比值大于 0.7），说明该特征点在目标图像中找到了多个相似度极高的候选点，极易发生误匹配；只有当比值严格小于 0.7 时，才认为这是一个唯一且可靠的匹配点对（即 good_matches）。

> 匹配数据统计：
> 从实验运行结果来看，两幅图像初始产生了 819 对粗匹配点。经过 Lowe's 比率测试的严苛筛选后，过滤掉了 612 对不可靠的匹配点，最终保留了 207 对高质量的优质匹配点，优质匹配点保留比例约为 25.27%。这一比例说明两幅图像存在充足的重叠区域。

```python
# 初始化SIFT检测器并提取特征 (这是你的 des_a 和 des_b 的来源)
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None) 
kp_b, des_b = sift.detectAndCompute(gray_b, None)

# 使用FLANN匹配器进行特征匹配 (你截图里的代码)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2) 

# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 输出匹配统计信息 (你截图里的代码)
print("=== 特征匹配统计 ===")
print(f"初始匹配点对数: {len(matches)}")
print(f"筛选后优质匹配点对数: {len(good_matches)}")
print(f"过滤掉的匹配点对数: {len(matches) - len(good_matches)}")
print(f"优质匹配点比例: {(len(good_matches)/len(matches)):.2%}")
```
输出:
<div align="center">
  <img alt="matched statistical information_image" src="https://github.com/user-attachments/assets/45cb934a-807c-47e6-a742-38476c71c654" />
</div>

> 匹配结果可视化：
> 通过 cv2.drawMatches 函数，将筛选出的 207 对优质匹配点绘制成连线图。从可视化结果可以看出，特征连线主要集中在建筑物特有的结构边缘（如窗户、线条、圆弧轮廓），排列整齐且无明显交叉错误，证明预筛选机制成功剔除了杂乱背景的干扰，为后续计算单应性矩阵提供了极其精确的坐标数据。

```python
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches,
    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 因为 OpenCV 画出来的图默认是 BGR 格式，而 matplotlib 需要 RGB 格式
matched_keypoints_img_rgb = cv2.cvtColor(matched_keypoints_img, cv2.COLOR_BGR2RGB)

# 创建一个画布，设置宽一点（因为拼接图通常很长）
plt.figure(figsize=(15, 8))

# 显示图片
plt.imshow(matched_keypoints_img_rgb)
plt.title("Matched SIFT Keypoints")
plt.axis('off') # 隐藏坐标轴
plt.show()      # 真正输出图片
```

输出：
<div align="center">
  <img alt="feature matching and connection diagram" src="https://github.com/user-attachments/assets/6574d682-faf1-4878-9005-c1434c08c14a" />
</div>

### 4. 计算单应性矩阵（图像配准）
- 坐标提取：从 good_matches 中分别提取图 B (源) 和图 A (基准) 的像素坐标。
- 求解单应性：利用 RANSAC 算法（随机抽样一致性）求解 3×3 的单应性矩阵 H。该算法能有效抵抗少数误匹配点的干扰，计算出透视映射关系。
- 画布计算与平移：将图 B 的四个顶点代入矩阵 H 计算变换后坐标，结合图 A 的尺寸确定最终全景画布的最大/最小边界。由于变换后坐标可能出现负数，构建了一个平移矩阵（translation_matrix），确保所有像素落在正向坐标系内。
- 图像融合：通过 cv2.warpPerspective 结合平移矩阵与矩阵 H，对图 B 进行形变。最后将图 A 按照计算好的位置复制到画布上，完成基础的空间对齐与融合。

```python
# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像B的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像A的关键点

# 使用RANSAC算法估计单应矩阵（透视变换矩阵）
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 输出单应性矩阵 H
print("【1】单应性矩阵 (H) - 描述图B到图A的透视变换：")
print(H)
print("-" * 50)

# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 计算图像变换后的四个角坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸（包含所有像素）
all_corners = np.concatenate([dst_corners, np.float32([[0,0], [w_a,0], [w_a,h_a], [0,h_a]]).reshape(-1,1,2)], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() + 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 创建平移矩阵，确保所有像素都在可视区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# 输出平移矩阵与画布尺寸
print("【2】平移矩阵 (Translation Matrix) - 用于将负坐标移回正区间：")
print(translation_matrix)
print(f"\n拼接画布尺寸 (宽 x 高): {(x_max - x_min)} x {(y_max - y_min)}")
print(f"左上角偏移量 (x_min, y_min): ({x_min}, {y_min})")
print("-" * 50)

# 组合平移矩阵和单应矩阵
combined_matrix = translation_matrix @ H

# 输出最终作用于图像的组合矩阵
print("【3】组合矩阵 (Translation @ H) - 最终直接用于图像B的变换：")
print(combined_matrix)
print("=" * 50)

# 对图像进行透视变换和平移
fus_img = cv2.warpPerspective(
    img_b,
    combined_matrix,  # 使用组合矩阵
    (x_max - x_min, y_max - y_min)  # 输出图像尺寸
)

# 将图像复制到拼接结果的对应位置
fus_img[-y_min : h_a - y_min, -x_min : w_a - x_min] = img_a

# 因为 OpenCV 存的是 BGR 格式，而 matplotlib 需要 RGB 格式，所以需要转换
fus_img_rgb = cv2.cvtColor(fus_img, cv2.COLOR_BGR2RGB)

# 创建画布并显示
plt.figure(figsize=(12, 8))
plt.imshow(fus_img_rgb)
plt.title('Stitched Result')
plt.axis('off')  # 隐藏坐标轴
plt.show()
```

输出：
<div align="center">
  <img alt="matrixs and concatenation result" src="https://github.com/user-attachments/assets/4573f679-d8ca-4b3d-9835-6b79bd9e8948" />
</div>
5. 图像透视变换与拼接
利用单应性矩阵 H，计算图像 B 变换后的四个角点坐标，并结合图像 A 的尺寸，确定最终拼接画布的大小。

计算平移矩阵（translation_matrix），确保所有像素都平移至可见的正坐标区域内。

使用 cv2.warpPerspective 将图像 B 进行透视变换及平移，映射到画布上。

将图像 A 复制覆盖到拼接结果图像的对应位置上，完成图像拼接。

6. 结果输出与展示
使用 cv2.imshow 显示特征点匹配图（Matched SIFT Keypoints）和最终拼接结果图（Stitched Result）。

使用 cv2.waitKey 和 cv2.destroyAllWindows 实现窗口交互与关闭。
