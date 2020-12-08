# Data analysis

Remote sensing images are often composed of many wavelengths, and the distribution of data in different wavelengths may vary greatly, for example, the visible and thermal infrared wavelengths are differently distributed. In order to better understand the distribution of the data to optimize the training effect of the model, the data needs to be analyzed.

## Statistical analysis
Run the following script to statistically analyze the training set. The analysis results are displayed in the screen. The results will be saved in `train_information.pkl`.

```
python tools/analysis.py
```

The statistical analysis of the data is as follows.

* Number of images

For example, 64 pictures are counted in the training set.
```
64 samples in file dataset/remote_sensing_seg/train.txt
```
* Maximum and minimum image size

For example, the maximum and minimum height and width of the training set are (1000, 1000) and (1000, 1000) respectively:
```
Minimal image height: 1000 Minimal image width:1000.
Maximal image height: 1000 Maximal image width:1000.
```
* Number of image channels

For example, the number of image channels is 10:

```
Image channel is 10.
```
* The minimum and maximum values for each image channel.

The minimum and maximum values are output as a list, arranged by channel in ascending order. For example:

```
Minimal image value: [7.172e+03 6.561e+03 5.777e+03 5.103e+03 4.291e+03 1.000e+00 1.000e+00 4.232e+03 6.934e+03 7.199e+03]
Maximal image value: [65535. 65535. 65535. 65535. 65535. 65535. 65535. 56534. 65535. 63215.]
```
* Distribution of pixel values of each image channel

For each channel, the number of pixel values is counted and presented as a histogram in the image ending with 'distribute.png'. **It should be noted that the vertical coordinates are logarithmic coordinates**. You can view these images and choose whether or not to clip the values of pixels distributed at the head and tail.

```
Image pixel distribution of each channel is saved with 'distribute.png' in the dataset/remote_sensing_seg
```

* Mean and variance normalized for each channel of the image.

The normalization factor for each channel is the difference between the maximum and minimum values in each channel, and the mean and variance are output in column form, arranged by channel in the ascending order. For example:

```
Image mean value: [0.23417574 0.22283101 0.2119595  0.2119887  0.27910388 0.21294892 0.17294037 0.10158925 0.43623915 0.41019192]
Image standard deviation: [0.06831269 0.07243951 0.07284761 0.07875261 0.08120818 0.0609302 0.05110716 0.00696064 0.03849307 0.03205579]
```

* Number and weight of categories on the chart

Statistics on the number of pixels in each category and the percentage of all pixels in the dataset, output in the format (category value, number of pixels of that category, percentage of pixels in that category). For example:

```
Label pixel information is shown in a format of (label_id, the number of label_id, the ratio of label_id):
(0, 13302870, 0.20785734374999995)
(1, 4577005, 0.07151570312500002)
(2, 3955012, 0.0617970625)
(3, 2814243, 0.04397254687499999)
(4, 39350870, 0.6148573437500001)
```

## 2. Determine the clipping range of the pixel value

Remote sensing image is widely distributed, with certain abnormal values. This affects the algorithm's fit to the actual data distribution. In order to better normalize the data, a small number of abnormal values in remote sensing images can be suppressed. The clip range of pixel values is determined based on the `pixel value distribution of each image channel`, and the out-of-range pixel values are corrected by clipping during subsequent image pre-processing to remove the interference caused by the abnormal values. **Note: This step is determined by the actual distribution of the data set.**

For example, the pixel value distribution for each channel is visualized as follows.

![](../../../examples/multi-channel_remote_sensing/docs/images/image_pixel_distribution.png)
**It should be noted that the vertical coordinates are logarithmic coordinates.**


For the above distribution, we have chosen the following truncation ranges (arranged according to channels from smallest to largest).

```
Truncation range min: clip_min_value = [7172, 6561, 5777, 5103, 4291, 4000, 4000, 4232, 6934, 7199]
Truncation range max: clip_max_value = [50000, 50000, 50000, 50000, 50000, 40000, 30000, 18000, 40000, 36000]
```

## 3 Determining the pixel value truncation range

In order to avoid the effects of incorrectly selected data truncation ranges, the percentage of outlier pixels should be counted to ensure that the proportion of affected pixels is not too high. Then calculate the normalized mean and variance of the truncated data **ifor mage pre-processing parameter settings for subsequent model training**.

Run the following scripts:
```
python tools/cal_clipped_mean_std.py
```

The statistics of the percentage of the clipped pixels are as follows:

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
It can be seen that the percentage of the clipped pixel does not exceed 0.2%.

The normalization coefficients for the clipped data are as follows.
```
Image mean value: [0.15163569 0.15142828 0.15574491 0.1716084  0.2799778  0.27652043 0.28195933 0.07853807 0.56333154 0.5477584 ]
Image standard deviation: [0.09301891 0.09818967 0.09831126 0.1057784  0.10842132 0.11062996 0.12791838 0.02637859 0.0675052  0.06168227]
(normalized by (clip_max_value - clip_min_value), arranged in 0-10 channel order)
```
