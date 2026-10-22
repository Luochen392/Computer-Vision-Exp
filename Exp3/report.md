# 实验三：图像关键点检测及图像拼接
202410315051-人工智能242-周落尘
## 一、实验目的
基于 SIFT 特征检测算法和 FLANN 特征匹配方法，实现两幅图像的特征点提取、匹配和自动拼接。掌握图像配准（单应性矩阵计算）和透视变换融合的基本原理与实现方法。
## 二、实验内容
导入必要库
cv2 (OpenCV库)：用于图像读取、SIFT特征提取、FLANN匹配、单应性矩阵计算及透视变换等核心操作。

numpy (Numpy库)：用于构建匹配点坐标矩阵、计算图像角点以及构建平移矩阵。

2. 图像预处理与特征提取
读取原图 a.jpg 和 b.jpg。

将彩色图像转换为灰度图（cv2.cvtColor），以减少计算量。

初始化 SIFT 检测器（cv2.SIFT_create），分别提取图像 A 和图像 B 的关键点（Keypoints）和特征描述符（Descriptors）。

3. 特征匹配与筛选
构建 FLANN 匹配器（cv2.FlannBasedMatcher），使用 KNN 算法（knnMatch，k=2）进行特征匹配。

应用 Lowe's 比率测试（Ratio Test），通过比较最近邻与次近邻的距离（阈值取 0.7），筛选出优质的匹配点对（good_matches）。

使用 cv2.drawMatches 绘制匹配的 SIFT 关键点连线图，用于直观验证匹配效果。

4. 计算单应性矩阵（图像配准）
从筛选后的优质匹配点中，分别提取图像 B（src_pts）和图像 A（dst_pts）的特征点坐标。

使用 RANSAC 算法（cv2.findHomography）估算两幅图像之间的单应性矩阵 H（透视变换矩阵）。

5. 图像透视变换与拼接
利用单应性矩阵 H，计算图像 B 变换后的四个角点坐标，并结合图像 A 的尺寸，确定最终拼接画布的大小。

计算平移矩阵（translation_matrix），确保所有像素都平移至可见的正坐标区域内。

使用 cv2.warpPerspective 将图像 B 进行透视变换及平移，映射到画布上。

将图像 A 复制覆盖到拼接结果图像的对应位置上，完成图像拼接。

6. 结果输出与展示
使用 cv2.imshow 显示特征点匹配图（Matched SIFT Keypoints）和最终拼接结果图（Stitched Result）。

使用 cv2.waitKey 和 cv2.destroyAllWindows 实现窗口交互与关闭。
