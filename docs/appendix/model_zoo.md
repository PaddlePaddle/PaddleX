# PaddleX模型库

## 图像分类模型
> 表中模型相关指标均为在ImageNet数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla P40），预测速度为每张图片预测用时（不包括预处理和后处理），表中符号`-`表示相关指标暂未测试。


| 模型  | 模型大小 | 预测速度（毫秒） | Top1准确率（%） | Top5准确率（%） |
| :----|  :------- | :----------- | :--------- | :--------- |
| [ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar)| 46.2MB   | 3.72882        | 71.0     | 89.9     |
| [ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar)| 87.9MB   | 5.50876        | 74.6    | 92.1    |
| [ResNet50](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar)| 103.4MB  | 7.76659       | 76.5     | 93.0     |
| [ResNet101](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) |180.4MB  | 13.80876      | 77.6     | 93.6  |
| [ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) |103.5MB  | 8.20476       | 79.1     | 94.4     |
| [ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar)| 180.5MB  | 14.24643       | 80.2   | 95.0     |
| [ResNet50_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar) |103.5MB  | 7.79264       | 82.4     | 96.1     |
| [ResNet101_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar)| 180.5MB  | 13.34580       | 83.7   | 96.7     |
| [DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar)|167.4MB  | 8.82047       | 78.0     | 94.1     |
| [MobileNetV1](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 17.4MB   | 3.42838        | 71.0     | 89.7    |
| [MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 15.0MB   | 5.92667        | 72.2     | 90.7    |
| [MobileNetV3_large](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar)|  22.8MB   | 8.31428        | 75.3    | 93.2   |
| [MobileNetV3_small](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar) |  12.5MB   | 7.30689        | 68.2    | 88.1     |
| [MobileNetV3_large_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar)|  22.8MB   | 8.06651        | 79.0     | 94.5     |
| [MobileNetV3_small_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_ssld_pretrained.tar) |  12.5MB   | 7.08837        | 71.3     | 90.1     |
| [Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) | 109.2MB   | 8.15611      | 79.6    | 94.4     |
| [Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) | 161.6MB  | 13.87017       | 80.3     | 94.5     |
| [DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | 33.1MB   | 17.09874       | 75.7     | 92.6     |
| [DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar)| 118.0MB  | 22.79690       | 78.6     | 94.1     |
| [DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar)|  84.1MB   | 25.26089       | 77.6     | 93.7     |
| [ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 10.2MB   | 15.40138        | 68.8     | 88.5     |
| [HRNet_W18](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar) | 21.29MB |45.25514  | 76.9 | 93.4 |
| [AlexNet](https://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 244.4MB | - | 56.7 | 79.2 |

## 目标检测模型

> 表中模型相关指标均为在MSCOCO数据集上使用PaddlePaddle Python预测接口测试得到（测试GPU型号为Nvidia Tesla V100测试得到），表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | BoxAP（%） |
|:-------|:-----------|:-------------|:----------|
|[FasterRCNN-ResNet18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/faster_rcnn_r18_fpn_1x.tar) | 173.2M | - | 32.6 |
|[FasterRCNN-ResNet50](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar)|136.0MB| 197.715 | 35.2 |
|[FasterRCNN-ResNet50_vd](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar)| 136.1MB | 475.700 | 36.4 |
|[FasterRCNN-ResNet101](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar)| 212.5MB | 582.911 | 38.3 |
|[FasterRCNN-ResNet50-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar)| 167.7MB | 83.189 | 37.2 |
|[FasterRCNN-ResNet50_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar)|167.8MB | 128.277 | 38.9 |
|[FasterRCNN-ResNet101-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar)| 244.2MB | 119.788 | 38.7 |
|[FasterRCNN-ResNet101_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) |244.3MB | 156.097 | 40.5 |
|[FasterRCNN-HRNet_W18-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_1x.tar) |115.5MB | 81.592 | 36 |
|[PPYOLO](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_2x.pdparams) | 329.1MB | - |45.9 |
|[YOLOv3-DarkNet53](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar)|249.2MB | 42.672 | 38.9 |
|[YOLOv3-MobileNetV1](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |99.2MB | 15.442 | 29.3 |
|[YOLOv3-MobileNetV3_large](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams)|100.7MB | 143.322 | 31.6 |
| [YOLOv3-ResNet34](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar)|170.3MB | 23.185 | 36.2 |

## 实例分割模型

> 预测时间是在一张Nvidia Tesla V100的GPU上通过'evaluate()'接口测试MSCOCO验证集得到，包括数据加载、网络前向执行和后处理, batch size是1，表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | BoxAP (%) | MaskAP (%)  |
|:-------|:-----------|:-------------|:----------|:----------|
|[MaskRCNN-ResNet18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_r18_fpn_1x.tar) | 189.1MB | - | 33.6 | 30.5 |
|[MaskRCNN-ResNet50](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) | 143.9MB | 87 | 38.2  | 33.4 |
|[MaskRCNN-ResNet50-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar)| 177.7MB | 63.9 | 38.7 | 34.7 |
|[MaskRCNN-ResNet50_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) | 177.7MB | 63.1 | 39.8 | 35.4 |
|[MaskRCNN-ResNet101-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) | 253.6MB | 77 | 39.5 | 35.2 |
|[MaskRCNN-ResNet101_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) | 253.7MB | 76.4 | 41.4 | 36.8 |
|[MaskRCNN-HRNet_W18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_hrnetv2p_w18_2x.tar) | 120.7MB | - | 38.7 | 34.7 |


## 语义分割模型

> 以下指标均在MSCOCO验证集上测试得到，表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | mIoU（%） |
|:-------|:-----------|:-------------|:----------|
|[DeepLabv3_MobileNetV2_x1.0](https://bj.bcebos.com/v1/paddleseg/deeplab_mobilenet_x1_0_coco.tgz)| 14.7MB | - | - |
|[DeepLabv3_Xception65](https://paddleseg.bj.bcebos.com/models/xception65_coco.tgz)| 329.3MB | - | - |
|[UNet](https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz) | 107.3MB | - | - |


> 以下指标均在Cityscapes验证集上测试得到，表中符号`-`表示相关指标暂未测试。

| 模型    | 模型大小    | 预测时间(毫秒) | mIoU（%） |
|:-------|:-----------|:-------------|:----------|
| [DeepLabv3_MobileNetV3_large_x1_0_ssld](https://paddleseg.bj.bcebos.com/models/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz) | 9.3MB | - | 73.28 |
| [DeepLabv3_MobileNetv2_x1.0](https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz) | 14.7MB | - | 69.8 |
| [DeepLabv3_Xception65](https://paddleseg.bj.bcebos.com/models/xception65_bn_cityscapes.tgz) | 329.3MB | - | 79.3 |
| [HRNet_W18](https://paddleseg.bj.bcebos.com/models/hrnet_w18_bn_cityscapes.tgz) | 77.3MB |  | 79.36 |
| [Fast-SCNN](https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar) | 9.8MB |  | 69.64 |
