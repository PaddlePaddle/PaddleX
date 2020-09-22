# 数据分析

遥感影像往往由许多波段组成，不同波段数据分布可能大相径庭，例如可见光波段和热红外波段分布十分不同。为了更深入了解数据的分布来优化模型训练效果，需要对数据进行分析。

## 统计分析
执行以下脚本，对训练集进行统计分析，屏幕会输出分析结果，同时结果也会保存至文件`train_information.pkl`中：

```
python tools/analysis.py
```

数据统计分析内容如下：

* 图像数量

例如统计出训练集中有64张图片：
```
64 samples in file dataset/remote_sensing_seg/train.txt
```
* 图像最大和最小的尺寸

例如统计出训练集中最大的高宽和最小的高宽分别是(1000, 1000)和(1000, 1000):
```
Minimal image height: 1000 Minimal image width: 1000.
Maximal image height: 1000 Maximal image width: 1000.
```
* 图像通道数量

例如统计出图像的通道数量为10:

```
Image channel is 10.
```
* 图像各通道的最小值和最大值

最小值和最大值分别以列表的形式输出，按照通道从小到大排列。例如：

```
Minimal image value: [7.172e+03 6.561e+03 5.777e+03 5.103e+03 4.291e+03 1.000e+00 1.000e+00 4.232e+03 6.934e+03 7.199e+03]
Maximal image value: [65535. 65535. 65535. 65535. 65535. 65535. 65535. 56534. 65535. 63215.]

```
* 图像各通道的像素值分布

针对各个通道，统计出各像素值的数量，并以柱状图的形式呈现在以'distribute.png'结尾的图片中。**需要注意的是，为便于观察，纵坐标为对数坐标**。用户可以查看这些图片来选择是否需要对分布在头部和尾部的像素值进行截断。

```
Image pixel distribution of each channel is saved with 'distribute.png' in the dataset/remote_sensing_seg
```

* 图像各通道归一化后的均值和方差

各通道归一化系数为各通道最大值与最小值之差，均值和方差以列别形式输出，按照通道从小到大排列。例如：

```
Image mean value: [0.23417574 0.22283101 0.2119595  0.2119887  0.27910388 0.21294892 0.17294037 0.10158925 0.43623915 0.41019192]
Image standard deviation: [0.06831269 0.07243951 0.07284761 0.07875261 0.08120818 0.0609302 0.05110716 0.00696064 0.03849307 0.03205579]
```

* 标注图中各类别的数量及比重

统计各类别的像素数量和在数据集全部像素的占比，以（类别值，该类别的数量，该类别的占比）的格式输出。例如：

```
Label pixel information is shown in a format of (label_id, the number of label_id, the ratio of label_id):
(0, 13302870, 0.20785734374999995)
(1, 4577005, 0.07151570312500002)
(2, 3955012, 0.0617970625)
(3, 2814243, 0.04397254687499999)
(4, 39350870, 0.6148573437500001)

```

## 2 确定像素值截断范围

遥感影像数据分布范围广，往往存在一些异常值，这会影响算法对实际数据分布的拟合效果。为更好地对数据进行归一化，可以抑制遥感影像中少量的异常值。根据`图像各通道的像素值分布`来确定像素值的截断范围，并在后续图像预处理过程中对超出范围的像素值通过截断进行校正，从而去除异常值带来的干扰。**注意：该步骤是否执行根据数据集实际分布来决定。**

例如各通道的像素值分布可视化效果如下：

![](../../../examples/multi-channel_remote_sensing/docs/images/image_pixel_distribution.png)
**需要注意的是，为便于观察，纵坐标为对数坐标。**


对于上述分布，我们选取的截断范围是(按照通道从小到大排列)：

```
截断范围最小值： clip_min_value = [7172,  6561,  5777, 5103, 4291, 4000, 4000, 4232, 6934, 7199]
截断范围最大值： clip_max_value = [50000, 50000, 50000, 50000, 50000, 40000, 30000, 18000, 40000, 36000]
```

## 3 确定像素值截断范围

为避免数据截断范围选取不当带来的影响，应该统计异常值像素占比，确保受影响的像素比例不要过高。接着对截断后的数据计算归一化后的均值和方差，**用于后续模型训练时的图像预处理参数设置**。

执行以下脚本：
```
python tools/cal_clipped_mean_std.py
```

截断像素占比统计结果如下:

```
Channel 0, the ratio of pixels to be clipped = 0.00054778125
Channel 1, the ratio of pixels to be clipped = 0.0011129375
Channel 2, the ratio of pixels to be clipped = 0.000843703125
Channel 3, the ratio of pixels to be clipped = 0.00127125
Channel 4, the ratio of pixels to be clipped = 0.001330140625
Channel 5, the ratio of pixels to be clipped = 8.1375e-05
Channel 6, the ratio of pixels to be clipped = 0.0007348125
Channel 7, the ratio of pixels to be clipped = 6.5625e-07
Channel 8, the ratio of pixels to be clipped = 0.000185921875
Channel 9, the ratio of pixels to be clipped = 0.000139671875
```
可看出，被截断像素占比均不超过0.2%。

裁剪后数据的归一化系数如下：
```
Image mean value: [0.15163569 0.15142828 0.15574491 0.1716084  0.2799778  0.27652043 0.28195933 0.07853807 0.56333154 0.5477584 ]
Image standard deviation: [0.09301891 0.09818967 0.09831126 0.1057784  0.10842132 0.11062996 0.12791838 0.02637859 0.0675052  0.06168227]
(normalized by (clip_max_value - clip_min_value), arranged in 0-10 channel order)
```
