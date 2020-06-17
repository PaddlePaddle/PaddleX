# PaddleX视觉方案介绍  

PaddleX目前提供了4种视觉任务解决方案，分别为图像分类、目标检测、实例分割和语义分割。用户可以根据自己的任务类型按需选取。

## 图像分类
图像分类任务指的是输入一张图片，模型预测图片的类别，如识别为风景、动物、车等。

![](./images/image_classification.png)

对于图像分类任务，针对不同的应用场景，PaddleX提供了百度改进的模型，见下表所示

|    模型    | 模型大小 | GPU预测速度 | CPU预测速度 | ARM芯片预测速度 | 准确率 | 备注 |
| :--------- | :------  | :---------- | :-----------| :-------------  | :----- | :--- |
| MobileNetV3_small_ssld | 12M | ? | ? | ? | 71.3% |适用于移动端场景 |
| MobileNetV3_large_ssld | 21M | ? | ? | ? | 79.0% | 适用于移动端/服务端场景 |
| ResNet50_vd_ssld | 102.8MB | ? | ? | ? | 82.4% | 适用于服务端场景 |
| ResNet101_vd_ssld | 179.2MB | ? | ? | ? |83.7% | 适用于服务端场景 |

除上述模型外，PaddleX还支持近20种图像分类模型，模型列表可参考[PaddleX模型库](../appendix/model_zoo.md)


## 目标检测
目标检测任务指的是输入图像，模型识别出图像中物体的位置（用矩形框框出来，并给出框的位置），和物体的类别，如在手机等零件质检中，用于检测外观上的瑕疵等。

![](./images/object_detection.png)

对于目标检测，针对不同的应用场景，PaddleX提供了主流的YOLOv3模型和Faster-RCNN模型，见下表所示

|   模型   | 模型大小  | GPU预测速度 | CPU预测速度 |ARM芯片预测速度 | BoxMAP | 备注 |
| :------- | :-------  | :---------  | :---------- | :-------------  | :----- | :--- |
| YOLOv3-MobileNetV1 | 101.2M | ? | ? | ? | 29.3 | |
| YOLOv3-MobileNetV3 | 94.6M | ? | ? | ? | 31.6 | |
| YOLOv3-ResNet34 | 169.7M | ? | ? | ? | 36.2 | |
| YOLOv3-DarkNet53 | 252.4 | ? | ? | ? | 38.9 | |

除YOLOv3模型外，PaddleX同时也支持FasterRCNN模型，支持FPN结构和5种backbone网络，详情可参考[PaddleX模型库](../appendix/model_zoo.md)

## 实例分割
在目标检测中，模型识别出图像中物体的位置和物体的类别。而实例分割则是在目标检测的基础上，做了像素级的分类，将框内的属于目标物体的像素识别出来。

![](./images/instance_segmentation.png)

PaddleX目前提供了实例分割MaskRCNN模型，支持5种不同的backbone网络，详情可参考[PaddleX模型库](../appendix/model_zoo.md)

|  模型 | 模型大小 | GPU预测速度 | CPU预测速度 | ARM芯片预测速度 | BoxMAP | SegMAP | 备注 |
| :---- | :------- | :---------- | :---------- | :-------------  | :----- | :----- | :--- |
| MaskRCNN-ResNet50_vd-FPN | 185.5M | ? | ? | ? | 39.8 | 35.4 | |
| MaskRCNN-ResNet101_vd-FPN | 268.6M | ? | ? | ? | 41.4 | 36.8 | |


## 语义分割
语义分割用于对图像做像素级的分类，应用在人像分类、遥感图像识别等场景。  

![](./images/semantic_segmentation.png)

对于语义分割，PaddleX也针对不同的应用场景，提供了不同的模型选择，如下表所示

| 模型 | 模型大小 | GPU预测速度 | CPU预测速度 | ARM芯片预测速度 | mIOU | 备注 |
| :---- | :------- | :---------- | :---------- | :-------------  | :----- | :----- |
| DeepLabv3p-MobileNetV2_x0.25 | | ? | ? | ? | ? | ? |
| DeepLabv3p-MobileNetV2_x1.0 | | ? | ? | ? | ? | ? |
| DeepLabv3p-Xception65 | | ? | ? | ? | ? | ? |
| UNet | | ? | ? | ? | ? | ? |
