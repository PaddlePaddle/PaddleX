# PaddleX模型库

## 图像分类模型

> 表中模型准确率均为在ImageNet数据集上测试所得，表中符号`-`表示相关指标暂未测试，预测速度测试环境如下所示:

* CPU的评估环境基于骁龙855（SD855）。
* GPU评估环境基于T4机器，在FP32+TensorRT配置下运行500次测得（去除前10次的warmup时间）。

### 移动端系列

| 模型  | 模型大小 | SD855 time(ms) bs=1 | Top1准确率（%） | Top5准确率（%） |
| :----|  :------- | :----------- | :--------- | :--------- |
| [MobileNetV1](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 17.4MB   | 32.523048  | 71.0     | 89.7    |
| [MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 15.0MB   | 23.317699  | 72.2     | 90.7    |
| [MobileNetV3_large](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar)|  22.8MB   | 19.30835  | 75.3    | 93.2   |
| [MobileNetV3_small](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar) |  12.5MB   | 9.2745  | 68.2    | 88.1     |
| [MobileNetV3_large_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar)|  22.8MB   | 19.30835 | 79.0     | 94.5     |
| [MobileNetV3_small_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_ssld_pretrained.tar) |  12.5MB   | 6.5463 | 71.3     | 90.1     |
| [ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 10.2MB   | 10.941        | 68.8     | 88.5     |

### 其他系列

| 模型  | 模型大小 | GPU time(ms) bs=1| Top1准确率（%） | Top5准确率（%） |
| :----|  :------- | :----------- | :--------- | :--------- |
| [ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar)| 46.2MB   | 1.45606       | 71.0     | 89.9     |
| [ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar)| 87.9MB   | 2.34957        | 74.6    | 92.1    |
| [ResNet50](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar)| 103.4MB  | 3.47712       | 76.5     | 93.0     |
| [ResNet101](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) |180.4MB  | 6.07125      | 77.6     | 93.6  |
| [ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) |103.5MB  | 3.53131       | 79.1     | 94.4     |
| [ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar)| 180.5MB  | 6.11704       | 80.2   | 95.0     |
| [ResNet50_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar) |103.5MB  | 3.53131       | 82.4     | 96.1     |
| [ResNet101_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar)| 180.5MB  | 6.11704       | 83.7   | 96.7     |
| [DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar)|167.4MB  | -       | 78.0     | 94.1     |
| [Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) | 109.2MB   | 4.96939      | 79.6    | 94.4     |
| [Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) | 161.6MB  | 7.26158       | 80.3     | 94.5     |
| [DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | 33.1MB   | 4.40447       | 75.7     | 92.6     |
| [DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar)| 118.0MB  | 10.39152       | 78.6     | 94.1     |
| [DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar)|  84.1MB   | 8.20652       | 77.6     | 93.7     |
| [HRNet_W18](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar) | 21.29MB | 7.40636  | 76.9 | 93.4 |
| [AlexNet](https://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 244.4MB | - | 56.7 | 79.2 |

## 目标检测模型

> 表中模型精度BoxAP通过`evaluate()`接口测试MSCOCO验证集得到，符号`-`表示相关指标暂未测试，预测时间在以下环境测试所的：

- 测试环境:
  - CUDA 9.0
  - CUDNN 7.5
  - PaddlePaddle v1.6
  - TensorRT-5.1.2.2
  - GPU分别为: Tesla V100
- 测试方式:
  - 为了方便比较不同模型的推理速度，输入采用同样大小的图片，为 3x640x640。
  - Batch Size=1
  - 去掉前10轮warmup时间，测试100轮的平均时间，单位ms/image，包括输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU的时间。
  - 采用Fluid C++预测引擎，开启FP32 TensorRT配置。
  - 测试时开启了 FLAGS_cudnn_exhaustive_search=True，使用exhaustive方式搜索卷积计算算法。

| 模型    | 模型大小    | 预测时间(ms/image) | BoxAP（%） |
|:-------|:-----------|:-------------|:----------|
|[FasterRCNN-ResNet18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/faster_rcnn_r18_fpn_1x.tar) | 173.2MB | - | 32.6 |
|[FasterRCNN-ResNet50](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar)|136.0MB| 146.124 | 35.2 |
|[FasterRCNN-ResNet50_vd](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar)| 136.1MB | 144.767 | 36.4 |
|[FasterRCNN-ResNet101](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar)| 212.5MB | 150.985 | 38.3 |
|[FasterRCNN-ResNet50-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar)| 167.7MB | 24.758 | 37.2 |
|[FasterRCNN-ResNet50_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar)|167.8MB | 25.292 | 38.9 |
|[FasterRCNN-ResNet101-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar)| 244.2MB | 30.331 | 38.7 |
|[FasterRCNN-ResNet101_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) |244.3MB | 29.969 | 40.5 |
|[FasterRCNN-HRNet_W18-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_1x.tar) |115.5MB | - | 36 |
|[PPYOLO](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_2x.pdparams) | 329.1MB | - |45.9 |
|[YOLOv3-DarkNet53](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar)|249.2MB | 20.252 | 38.9 |
|[YOLOv3-MobileNetV1](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) |99.2MB | 11.834 | 29.3 |
|[YOLOv3-MobileNetV3_large](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams)|100.7MB | - | 31.6 |
| [YOLOv3-ResNet34](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar)|170.3MB | 14.125 | 36.2 |

## 实例分割模型

> 表中模型精度BoxAP/MaskAP通过`evaluate()`接口测试MSCOCO验证集得到，符号`-`表示相关指标暂未测试，预测时间在以下环境测试所的

- 测试环境:
  - CUDA 9.0
  - CUDNN 7.5
  - PaddlePaddle v1.6
  - TensorRT-5.1.2.2
  - GPU分别为: Tesla V100
- 测试方式:
  - 为了方便比较不同模型的推理速度，输入采用同样大小的图片，为 3x640x640。
  - Batch Size=1
  - 去掉前10轮warmup时间，测试100轮的平均时间，单位ms/image，包括输入数据拷贝至GPU的时间、计算时间、数据拷贝至CPU的时间。
  - 采用Fluid C++预测引擎，开启FP32 TensorRT配置。
  - 测试时开启了 FLAGS_cudnn_exhaustive_search=True，使用exhaustive方式搜索卷积计算算法。

| 模型    | 模型大小    | 预测时间(毫秒) | BoxAP (%) | MaskAP (%)  |
|:-------|:-----------|:-------------|:----------|:----------|
|[MaskRCNN-ResNet18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_r18_fpn_1x.tar) | 189.1MB | - | 33.6 | 30.5 |
|[MaskRCNN-ResNet50](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) | 143.9MB | 159.527 | 38.2  | 33.4 |
|[MaskRCNN-ResNet50-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar)| 177.7MB | 83.567 | 38.7 | 34.7 |
|[MaskRCNN-ResNet50_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) | 177.7MB | 97.929 | 39.8 | 35.4 |
|[MaskRCNN-ResNet101-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) | 253.6MB | 97.929 | 39.5 | 35.2 |
|[MaskRCNN-ResNet101_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) | 253.7MB | 97.647 | 41.4 | 36.8 |
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
| [HRNet_W18](https://paddleseg.bj.bcebos.com/models/hrnet_w18_bn_cityscapes.tgz) | 77.3MB | - | 79.36 |
| [Fast-SCNN](https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar) | 9.8MB | - | 69.64 |
