# 实验⼆：图像增强参考⽂档
202410315051-人工智能242-周洛尘
## ⼀、实验⽬的
学会OpenCV的基本使⽤⽅法，利⽤OpenCV等计算机库对图像进⾏平滑、滤波等操作，实现图像增强。
## ⼆、实验内容
### 2.1 导⼊图像滤波相关的依赖包
- 1.OpenCV库：计算机视觉与图像处理
- 2.scikit-image库：图像增强与分析random_noise ⽤于给图像添加噪声
- 3.NumPy库：科学计算的基础库
- 4.Matplotlib库：绘图与数据可视化
```python
import cv2
from skimage.util import random_noise
import numpy as np
from matplotlib import pyplot as plt
```
### 2.2 读取原始图像并进⾏⾊彩空间转换
读取计算机本地图像⽂件，获取并输⼊【100，100】处像素点的RBG参数并输出,通过下⾯的输出结果可以看到，【100,100】像素点的RGB为【82,86,241】，表现为偏蓝⾊，通过图像也可以看到这个像素点属于猫猫的⾐服处的深蓝⾊部位
```python
img = cv2.imread('p1.jpg')
# 获取图像中【100，100】这个像素的rbg三⾊
(b, g, r) = img[100, 100]
# 打印这个像素点的rbg参数
print(b, g, r)
# 输出原始图像
plt.imshow(img)
plt.title('Original Image')
plt.savefig('output_images/original_bgr.jpg', dpi=300)
plt.show()
```
输出：
<div align="center">
  <img alt="Original_image" src="https://github.com/user-attachments/assets/c46d546b-3689-4eb5-9104-7376a2d9ee44" />
</div>
然后测试⼀下cv2中颜⾊空间变换的效果，这⾥的cv2.cvtColor就是颜⾊空间转换，cv2.COLOR_BGR2RGB代表的是将原始图像BGR格式转换成RGB格式，蓝⾊和红⾊互换，因为把'B'和'R'通道互换了，所以这是⼀个红蓝的颜⾊反转

```python
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title('RGB Image')
plt.savefig('output_images/rgb_image.jpg', dpi=300)
plt.show()
```
红蓝反准图：
<div align="center">
  <img alt="RGB_image" src="https://github.com/user-attachments/assets/7f589cd4-5d87-4a73-bfa9-71b0a8cbb7e2" />
</div>
cv2.COLOR_BGR2GRAY是将原始图像的RBG格式转换为灰度图，将三维的RGB通道映射为⼀维的灰度通道

```python
# 将原始图像的RBG格式转换为灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title('Gray Image')
plt.savefig('output_images/gray_image.jpg', dpi=300)
plt.show()
```
灰度图：
<div align="center">
  <img alt="Gray_image" src="https://github.com/user-attachments/assets/09b6a5c3-2e36-402d-9b12-4b41e5e52ac3" />
</div>

### 2.3 添加噪声
这⾥在原始图像的基础上添加噪声，引⼊了两个API⽅法，椒盐噪声和⾼斯噪声，通过对⽐可以发现，椒盐噪声和⾼斯噪声的本质不同，椒盐噪声表现为像素会随机替换为⽩⾊或者⿊⾊像素（灰度通道），在RGB通道表现为像素变成随机彩⾊点，⽽⾼斯噪声会在每个像素上添加随机偏差，服从⾼斯分布
```python
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.4)
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0.2, var=0.03)
# 原图
plt.subplot(1, 3, 1)
plt.imshow(rgb_img, cmap='gray')
plt.title('Original Image')
# 椒盐噪声
plt.subplot(1, 3, 2)
plt.imshow(sp_noise_img, cmap='gray')
plt.title('S&P Noise')
# ⾼斯噪声
plt.subplot(1, 3, 3)
plt.imshow(gus_noise_img, cmap='gray')
plt.title('Gus Noise')
plt.tight_layout()
plt.savefig('output_images/noise_comparison.jpg', dpi=300)
plt.show()
```
噪声对比图：
<div align="center">
  <img alt="Composed_images" src="https://github.com/user-attachments/assets/477fa8b4-29bd-4a3c-87ed-a4f28a2009a2" />
</div>

### 2.4 图像滤波
将图像认为产⽣噪声后，⽤OpenCV的三个API滤波⽅式进⾏对⽐，分别对椒盐滤波和⾼斯滤波使⽤【均值滤波】，【中值滤波】，【⾼斯滤波】，对⽐每个最适合的滤波⽅式。
```python
mean_sp = cv2.blur(sp_noise_img, (5, 5))
mean_gus = cv2.blur(gus_noise_img, (5, 5))

mid_sp = cv2.medianBlur((sp_noise_img*255).astype(np.uint8), 5)
mid_gus = cv2.medianBlur((gus_noise_img*255).astype(np.uint8), 5)

gauss_sp = cv2.GaussianBlur((sp_noise_img*255).astype(np.uint8), (5, 5), 0)
gauss_gus = cv2.GaussianBlur((gus_noise_img*255).astype(np.uint8), (5, 5), 0)

plt.figure(figsize=(13, 9))

plt.subplot(2, 3, 1)
plt.imshow(mean_sp)
plt.title("S&P noise with Mean Filter")
plt.subplot(2, 3, 2)
plt.imshow(mid_sp)
plt.title("S&P Noise with Median Filter")
plt.subplot(2, 3, 3)
plt.imshow(gauss_sp)
plt.title("S&P Noise with Gaussian Filter")

plt.subplot(2, 3, 4)
plt.imshow(mean_gus)
plt.title("Gaussian noise with Mean Filter")
plt.subplot(2, 3, 5)
plt.imshow(mid_gus)
plt.title("Gaussian noise with Median Filter")
plt.subplot(2, 3, 6)
plt.imshow(gauss_gus)
plt.title("Gaussian noise with Gaussian Filter")
plt.tight_layout()
plt.savefig('result/filter_results_2x3.jpg', dpi=300)
plt.show()
```
图像滤波对⽐图：
<div align="center">
  <img alt="Composed_images2" src="https://github.com/user-attachments/assets/672430f4-5fda-418f-aa31-2436fb2afab0" />
</div>

### 2.5 ⼿动实现⼀个滤波⽅式（中值滤波）

```python
manual_mid = cv2.medianBlur((sp_noise_img * 255).astype(np.uint8), 5)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sp_noise_img, cmap='gray')
plt.title("S&P noise img")
plt.subplot(1, 2, 2)
plt.imshow(manual_mid, cmap='gray')
plt.title("median filter")
plt.tight_layout()
plt.show()
```
⼿动构建中值滤波效果图：
<div align="center">
  <img alt="result_image" src="https://github.com/user-attachments/assets/53c7357d-3908-446a-b304-bdf371d9e94b" />
</div>

## 三、实验结果与分析
### 1.原始图像与颜⾊空间转换
- 通过 OpenCV 读取图像并显⽰，可以清楚看到原始彩⾊图像的细节。
- 将 BGR 格式转换为 RGB 格式后，颜⾊显⽰更符合⼈眼认知，红、绿、蓝通道正确对应。
- 灰度图显⽰则突出图像的亮度信息，有助于后续图像处理分析。
### 2.噪声添加效果
- 椒盐噪声（s&p noise）：在图像中随机出现⿊⽩点，使图像部分区域明显破坏。
- ⾼斯噪声（Gaussian noise）：整个图像亮度轻微抖动，更加均匀，整体细节略模糊。
- 对⽐实验显⽰，椒盐噪声的局部破坏更明显，⾼斯噪声则影响全局视觉效果。
### 3.滤波去噪效果
- 均值滤波（Mean Filter）：
  
  对⾼斯噪声的去除效果较好，能够平滑整幅图像，但对椒盐噪声中的尖锐⿊⽩点去除不彻底，且容易造成边缘模糊。
- 中值滤波（cv2.medianBlur）：
  
  对椒盐噪声的去除效果显著，能有效保留图像边缘和细节；对⾼斯噪声也有⼀定的抑制作⽤，但整体去噪效果略逊于均值滤波。
- ⾼斯滤波（cv2.GaussianBlur）：
  
  对⾼斯噪声的去除效果最佳，平滑⾃然，且保留了⼀定边缘信息；但对椒盐噪声的处理能⼒有限，因为其噪声是突变的⽽⾮连续分布。
- ⼿动实现中值滤波（彩⾊）：
  
  效果与 OpenCV ⾃带中值滤波基本⼀致，能有效去除彩⾊图像中的椒盐噪声，并保持颜⾊真实和边缘清晰。实验验证了对彩⾊图像应分别对 R、G、B 三通道独⽴滤波的合理性。通过对⽐显⽰，⼿动中值滤波在保持彩⾊信息、边缘清晰度⽅⾯表现良好，说明对彩⾊图像处理时，需要对每个通道分别滤波。
### 总体分析
不同类型的噪声适合采⽤不同的滤波⽅法进⾏去噪处理：
- 椒盐噪声（Salt & Pepper Noise） → 由于其属于突变型噪声，中值滤波（Median Filter） 能有效去除孤⽴的⿊⽩噪点，同时较好地保留图像边缘细节。
- ⾼斯噪声（Gaussian Noise） → 属于连续型噪声，适合采⽤均值滤波（Mean Filter）或⾼斯滤波（Gaussian Filter）进⾏平滑处理，能够在抑制噪声的同时保持较⾃然的视觉效果。
⼿动实现的彩⾊中值滤波不仅复现了 OpenCV 中的滤波效果，还加深了对滤波原理的理解。在实现过程中，通过分别对 R、G、B 三个通道独⽴处理，能够灵活调整滤波核⼤⼩和算法逻辑，为后续针对不同噪声类型的⾃定义滤波与优化提供了良好的基础。
## 四、实验⼩结
本实验完成了彩⾊图像的读取、BGR→RGB 转换以及灰度图显⽰，并在此基础上为图像添加了椒盐噪声与⾼斯噪声。通过分别采⽤均值滤波、中值滤波以及⼿动实现的中值滤波对噪声图像进⾏去噪处理，对⽐分析了不同滤波⽅法的效果。
实验结果表明：
- 中值滤波对椒盐噪声的去除效果最为显著，能够有效保留图像边缘和细节；
- 均值滤波更适⽤于⾼斯噪声的平滑去除；
- ⼿动实现的中值滤波在处理彩⾊图像时同样表现良好，既能有效抑制噪声，⼜能保持图像的真实⾊彩与结构信息。

通过本次实验，进⼀步加深了对图像噪声类型与滤波原理的理解，掌握了⼿动
实现彩⾊图像中值滤波的基本⽅法，为后续图像去噪与滤波算法的改进与优化
奠定了基础。
