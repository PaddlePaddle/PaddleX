# PaddleX模型库

## 图像分类模型
> 表中模型相关指标均为在ImageNet数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla P40），预测速度为每张图片预测用时（不包括预处理和后处理）,表中符号`-`表示相关指标暂未测试。


| 模型  | 模型大小 | 预测速度（毫秒） | Top1准确率（%） | Top5准确率（%） |
| :----|  :------- | :----------- | :--------- | :--------- |
| ResNet18| 46.2MB   | 3.72882        | 71.0     | 89.9     |
| ResNet34| 87.9MB   | 5.50876        | 74.6    | 92.1    |
| ResNet50| 103.4MB  | 7.76659       | 76.5     | 93.0     |
| ResNet101 |180.4MB  | 13.80876      | 77.6     | 93.6  |
| ResNet50_vd |103.5MB  | 8.20476       | 79.1     | 94.4     |
| ResNet101_vd| 180.5MB  | 14.24643       | 80.2   | 95.0     |
| ResNet50_vd_ssld |103.5MB  | 7.79264       | 82.4     | 96.1     |
| ResNet101_vd_ssld| 180.5MB  | 13.34580       | 83.7   | 96.7     |
| DarkNet53|167.4MB  | 8.82047       | 78.0     | 94.1     |
| MobileNetV1 | 17.4MB   | 3.42838        | 71.0     | 89.7    |
| MobileNetV2 | 15.0MB   | 5.92667        | 72.2     | 90.7    |
| MobileNetV3_large|  22.8MB   | 8.31428        | 75.3    | 93.2   |
| MobileNetV3_small |  12.5MB   | 7.30689        | 68.2    | 88.1     |
| MobileNetV3_large_ssld|  22.8MB   | 8.06651        | 79.0     | 94.5     |
| MobileNetV3_small_ssld |  12.5MB   | 7.08837        | 71.3     | 90.1     |
| Xception41 | 109.2MB   | 8.15611      | 79.6    | 94.4     |
| Xception65 | 161.6MB  | 13.87017       | 80.3     | 94.5     |
| DenseNet121 | 33.1MB   | 17.09874       | 75.7     | 92.6     |
| DenseNet161| 118.0MB  | 22.79690       | 78.6     | 94.1     |
| DenseNet201|  84.1MB   | 25.26089       | 77.6     | 93.7     |
| ShuffleNetV2 | 10.2MB   | 15.40138        | 68.8     | 88.5     |
| HRNet_W18 | 21.29MB |45.25514  | 76.9 | 93.4 |

## 目标检测模型

> 表中模型相关指标均为在MSCOCO数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla V100测试得到）,表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | BoxAP（%） |
|:-------|:-----------|:-------------|:----------|
|FasterRCNN-ResNet50|136.0MB| 197.715 | 35.2 |
|FasterRCNN-ResNet50_vd| 136.1MB | 475.700 | 36.4 |
|FasterRCNN-ResNet101| 212.5MB | 582.911 | 38.3 |
|FasterRCNN-ResNet50-FPN| 167.7MB | 83.189 | 37.2 |
|FasterRCNN-ResNet50_vd-FPN|167.8MB | 128.277 | 38.9 |
|FasterRCNN-ResNet101-FPN| 244.2MB | 156.097 | 38.7 |
|FasterRCNN-ResNet101_vd-FPN |244.3MB | 119.788 | 40.5 |
|FasterRCNN-HRNet_W18-FPN |115.5MB | 81.592 | 36 |
|YOLOv3-DarkNet53|249.2MB | 42.672 | 38.9 |
|YOLOv3-MobileNetV1 |99.2MB | 15.442 | 29.3 |
|YOLOv3-MobileNetV3_large|100.7MB | 143.322 | 31.6 |
| YOLOv3-ResNet34|170.3MB | 23.185 | 36.2 |

## 实例分割模型

> 表中模型相关指标均为在MSCOCO数据集上测试得到。

| 模型    | 模型大小    | 预测时间(毫秒) | mIoU（%） |
|:-------|:-----------|:-------------|:----------|
|DeepLabv3+-MobileNetV2_x1.0|-| - | - |
|DeepLabv3+-Xception41|-| - | - |
|DeepLabv3+-Xception65|-| - | - |
|UNet|-| - | - |
|HRNet_w18|-| - | - |
