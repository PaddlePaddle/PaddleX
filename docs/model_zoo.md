# 模型库
本文档梳理了PaddleX v0.1.0支持的模型，同时也提供了在各个数据集上的预训练模型和对应验证集上的指标。用户也可自行下载对应的代码，在安装PaddleX后，即可使用相应代码训练模型。

表中相关模型也可下载好作为相应模型的预训练模型，通过`pretrain_weights`指定目录加载使用。

## 图像分类模型
> 表中模型相关指标均为在ImageNet数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla P4），预测速度为每张图片预测用时（不包括预处理和后处理）,表中符号`-`表示相关指标暂未测试。


| 模型  | 模型大小 | 预测速度（毫秒） | Top1准确率 | Top5准确率 |
| :----|  :------- | :----------- | :--------- | :--------- |
| ResNet18| 46.9MB   | 3.456        | 70.98%     | 89.92%     |
| ResNet34| 87.5MB   | 5.668        | 74.57%     | 92.14%     |
| ResNet50| 102.7MB  | 8.787        | 76.50%     | 93.00%     |
| ResNet101 |179.1MB  | 15.447      | 77.56%     | 93.64%    |
| ResNet50_vd |102.8MB  | 9.058        | 79.12%     | 94.44%     |
| ResNet101_vd| 179.2MB  | 15.685       | 80.17%     | 94.97%     |
| DarkNet53|166.9MB  | 11.969       | 78.04%     | 94.05%     |
| MobileNetV1 | 16.4MB   | 2.609        | 70.99%     | 89.68%     |
| MobileNetV2 | 14.4MB   | 4.546        | 72.15%     | 90.65%     |
| MobileNetV3_large|  22.8MB   | -        | 75.3%     | 75.3%     |
| MobileNetV3_small |  12.5MB   | 6.809        | 67.46%     | 87.12%     |
| Xception41 |92.4MB   | 13.757       | 79.30%     | 94.53%     |
| Xception65 | 144.6MB  | 19.216       | 81.00%     | 95.49%     |
| Xception71| 151.9MB  | 23.291       | 81.11%     | 95.45%     |
| DenseNet121 | 32.8MB   | 12.437       | 75.66%     | 92.58%     |
| DenseNet161|116.3MB  | 27.717       | 78.57%     | 94.14%     |
| DenseNet201|  84.6MB   | 26.583       | 77.63%     | 93.66%     |
| ShuffleNetV2 | 10.2MB   | 6.101        | 68.8%     | 88.5%     |

## 目标检测模型

> 表中模型相关指标均为在MSCOCO数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla V100测试得到,表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | BoxAP |
|:-------|:-----------|:-------------|:----------|
|FasterRCNN-ResNet50|135.6MB| 78.450 | 35.2 |
|FasterRCNN-ResNet50_vd| 135.7MB | 79.523 | 36.4 |
|FasterRCNN-ResNet101| 211.7MB | 107.342 | 38.3 |
|FasterRCNN-ResNet50-FPN| 167.2MB | 44.897 | 37.2 |
|FasterRCNN-ResNet50_vd-FPN|168.7MB | 45.773 | 38.9 |
|FasterRCNN-ResNet101-FPN| 251.7MB | 55.782 | 38.7 |
|FasterRCNN-ResNet101_vd-FPN |252MB | 58.785 | 40.5 |
|YOLOv3-DarkNet53|252.4MB | 21.944 | 38.9 |
|YOLOv3-MobileNetv1 |101.2MB | 12.771 | 29.3 |
|YOLOv3-MobileNetv3|94.6MB | - | 31.6 |
| YOLOv3-ResNet34|169.7MB | 15.784 | 36.2 |

## 实例分割模型

> 表中模型相关指标均为在MSCOCO数据集上测试得到。

| 模型 |模型大小 | 预测时间(毫秒) | BoxAP | SegAP |
|:---------|:---------|:----------|:---------|:--------|
|MaskRCNN-ResNet50|51.2MB| 86.096 | 36.5 |32.2|
|MaskRCNN-ResNet50-FPN|184.6MB | 65.859 | 37.9 |34.2|
|MaskRCNN-ResNet50_vd-FPN |185.5MB | 63.191 | 39.8 |35.4|
|MaskRCNN-ResNet101-FPN|268.6MB | 77.024 | 39.5 |35.2|
|MaskRCNN-ResNet101vd-FPN |268.6MB | 76.307 | 41.4 |36.8|

## 语义分割模型

> 表中符号`-`表示相关指标暂未测试。

| 模型|数据集 | 模型大小 | 预测速度 | mIOU |
|:--------|:----------|:----------|:----------|:----------|
| UNet| | COCO | 53.7M | - |
| DeepLabv3+/Xception65| Cityscapes | 165.1M | | 0.7930 |
| DeepLabv3+/MobileNetV2 | Cityscapes | 7.4M |  | 0.6981 |
