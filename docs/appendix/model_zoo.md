# PaddleX模型库

## 图像分类模型
> 表中模型相关指标均为在ImageNet数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla P40），预测速度为每张图片预测用时（不包括预处理和后处理）,表中符号`-`表示相关指标暂未测试。


| 模型  | 模型大小 | 预测速度（毫秒） | Top1准确率（%） | Top5准确率（%） |
| :----|  :------- | :----------- | :--------- | :--------- |
| ResNet18| 46.9MB   | -        | 71.0     | 89.9     |
| ResNet34| 87.5MB   | -        | 74.6    | 92.1    |
| ResNet50| 102.7MB  | -        | 76.5     | 93.0     |
| ResNet101 |179.1MB  | -      | 77.6     | 93.6  |
| ResNet50_vd |102.8MB  |-        | 79.1     | 94.4     |
| ResNet101_vd| 179.2MB  | -       | 80.2   | 95.0     |
| ResNet50_vd_ssld |102.8MB  | -        | 82.4     | 96.1     |
| ResNet101_vd_ssld| 179.2MB  | -       | 83.7   | 96.7     |
| DarkNet53|166.9MB  | -       | 78.0     | 94.1     |
| MobileNetV1 | 16.0MB   | -        | 71.0     | 89.7    |
| MobileNetV2 | 14.0MB   | -        | 72.2     | 90.7    |
| MobileNetV3_large|  21.0MB   | -        | 75.3    | 93.2   |
| MobileNetV3_small |  12.0MB   | -        | 68.2    | 88.1     |
| MobileNetV3_large_ssld|  21.0MB   | -        | 79.0     | 94.5     |
| MobileNetV3_small_ssld |  12.0MB   | -        | 71.3     | 90.1     |
| Xception41 |92.4MB   | -       | 79.6    | 94.4     |
| Xception65 | 144.6MB  | -       | 80.3     | 94.5     |
| DenseNet121 | 32.8MB   | -       | 75.7     | 92.6     |
| DenseNet161|116.3MB  | -       | 78.6     | 94.1     |
| DenseNet201|  84.6MB   | -       | 77.6     | 93.7     |
| ShuffleNetV2 | 9.0MB   | -        | 68.8     | 88.5     |
| HRNet_W18 | 21.29MB | - | 76.9 | 93.4 |

## 目标检测模型

> 表中模型相关指标均为在MSCOCO数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla V100测试得到,表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | BoxAP（%） |
|:-------|:-----------|:-------------|:----------|
|FasterRCNN-ResNet50|135.6MB| 78.450 | 35.2 |
|FasterRCNN-ResNet50_vd| 135.7MB | 79.523 | 36.4 |
|FasterRCNN-ResNet101| 211.7MB | 107.342 | 38.3 |
|FasterRCNN-ResNet50-FPN| 167.2MB | 44.897 | 37.2 |
|FasterRCNN-ResNet50_vd-FPN|168.7MB | 45.773 | 38.9 |
|FasterRCNN-ResNet101-FPN| 251.7MB | 55.782 | 38.7 |
|FasterRCNN-ResNet101_vd-FPN |252MB | 58.785 | 40.5 |
|FasterRCNN-HRNet_W18-FPN |115.5MB | 57.11 | 36 |
|YOLOv3-DarkNet53|252.4MB | 21.944 | 38.9 |
|YOLOv3-MobileNetv1 |101.2MB | 12.771 | 29.3 |
|YOLOv3-MobileNetv3|94.6MB | - | 31.6 |
| YOLOv3-ResNet34|169.7MB | 15.784 | 36.2 |

## 实例分割模型

> 表中模型相关指标均为在MSCOCO数据集上测试得到。
