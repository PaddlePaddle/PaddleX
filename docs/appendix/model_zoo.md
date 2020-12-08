# PaddleX model library

## Image classification model

> The model accuracy rate in the table IS based on the ImageNet dataset, and the symbol `-` in the table indicates that the metric has not been tested yet:

* The CPU evaluation environment is based on Snapdragon 855 (SD855).
* The GPU evaluation environment is based on a T4 machine with 500 runs in FP32+TensorRT configuration (removing warmup time for the first 10 runs).

### Mobile Series

| Model | Model size | SD855 time(ms) bs=1 | Top 1 Accuracy rate (%) | Top 5 accuracy rate (%) |
| :----|  :------- | :----------- | :--------- | :--------- |
| [MobileNetV1](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | 17.4MB | 32.523048 | 71.0 | 89.7 |
| [MobileNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | 15.0MB | 23.317699 | 72.2 | 90.7 |
| [MobileNetV3_large](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar) | 22.8MB | 19.30835 | 75.3 | 93.2 |
| [MobileNetV3_small](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar) | 12.5MB | 9.2745 | 68.2 | 88.1 |
| [MobileNetV3_large_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar) | 22.8MB | 19.30835 | 79.0 | 94.5 |
| [MobileNetV3_small_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_ssld_pretrained.tar) | 12.5MB | 6.5463 | 71.3 | 90.1 |
| [ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | 10.2MB | 10.941 | 68.8 | 88.5 |

### Other series

| Model | Model size | GPU time(ms) bs=1 | Top 1 Accuracy rate (%) | Top 5 accuracy rate (%) |
| :----|  :------- | :----------- | :--------- | :--------- |
| [ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | 46.2MB | 1.45606 | 71.0 | 89.9 |
| [ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | 87.9MB | 2.34957 | 74.6 | 92.1 |
| [ResNet50](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | 103.4MB | 3.47712 | 76.5 | 93.0 |
| [ResNet101](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | 180.4MB | 6.07125 | 77.6 | 93.6 |
| [ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | 103.5MB | 3.53131 | 79.1 | 94.4 |
| [ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | 180.5MB | 6.11704 | 80.2 | 95.0 |
| [ResNet50_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar) | 103.5MB | 3.53131 | 82.4 | 96.1 |
| [ResNet101_vd_ssld](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar) | 180.5MB | 6.11704 | 83.7 | 96.7 |
| [DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar) | 167.4MB | -- | 78.0 | 94.1 |
| [Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) | 109.2MB | 4.96939 | 79.6 | 94.4 |
| [Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) | 161.6MB | 7.26158 | 80.3 | 94.5 |
| [densenet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | 33.1MB | 4.40447 | 75.7 | 92.6 |
| [densenet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar) | 118.0MB | 10.39152 | 78.6 | 94.1 |
| [densenet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar) | 84.1MB | 8.20652 | 77.6 | 93.7 |
| [HRNet_W18](https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar) | 21.29MB | 7.40636 | 76.9 | 93.4 |
| [AlexNet](https://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar) | 244.4MB | -- | 56.7 | 79.2 |

## Object detection model

> The model accuracy BoxAP in the table is obtained by testing the MSCOCO validation set through the `evaluate()` interface. The symbol `-` indicates that the relevant metrics have not been yet tested and the prediction time is obtained in the following environments:

- Test environment:
   - CUDA 9.0
   - CUDNN 7.5
   - PaddlePaddle v1.6
   - TensorRT-5.1.2.2
   - GPU: Tesla V100
- Test mode:
   - To make it easier to compare the reasoning speed of different models, the input is the same size as the image: 3x640x640.
   - Batch Size=1
   - Remove the first 10 rounds of warmup time and test the average time in the unit of ms/image for 100 rounds, including the time to copy the input data to the GPU, computation time, and time to copy data to CPU.
   - Use the Fluid C++ prediction engine. Enable the FP32 TensorRT configuration.
   - Start the test. FLAGS_cudnn_exhaustive_search=True: search for the convolutional algorithm using the exhaustive method.

| Model | Model size | Prediction time (ms/image) | BoxAP (%) |
|:-------|:-----------|:-------------|:----------|
| [FasterRCNN-ResNet18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/faster_rcnn_r18_fpn_1x.tar) | 173.2 MB | -- | 32.6 |
| [FasterRCNN-ResNet50](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_1x.tar) | 136.0MB | 146.124 | 35.2 |
| [FasterRCNN-ResNet50_vd](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_1x.tar) | 136.1MB | 144.767 | 36.4 |
| [FasterRCNN-ResNet101](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_1x.tar) | 212.5MB | 150.985 | 38.3 |
| [FasterRCNN-ResNet50-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_1x.tar) | 167.7 MB | 24.758 | 37.2 |
| [FasterRCNN-ResNet50_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar) | 167.8MB | 25.292 | 38.9 |
| [FasterRCNN-ResNet101-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_1x.tar) | 244.2MB | 30.331 | 38.7 |
| [FasterRCNN-ResNet101_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar) | 244.3MB | 29.969 | 40.5 |
| [FasterRCNN - HRNet_W18-FPN](https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_1x.tar) | 115.5MB | -- | 36 |
| [PPYOLO](https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_2x.pdparams) | 329.1 MB | -- | 45.9 |
| [YOLOv3-DarkNet53](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar) | 249.2 MB | 20.252 | 38.9 |
| [YOLOv3-MobileNetV1](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar) | 99.2 MB | 11.834 | 29.3 |
| [YOLOv3-MobileNetV3_large](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v3.pdparams) | 100.7 MB | -- | 31.6 |
| [YOLOv3-ResNet34](https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar) | 170.3MB | 14.125 | 36.2 |

## Instance segmentation model

> 表中模型精度BoxAP/MaskAP通过`evaluate()`接口测试MSCOCO验证集得到，符号`-`表示相关指标暂未测试，预测时间在以下环境测试所的 The model precision BoxAP/MaskAP in the table is obtained by testing the MSCOCO validation set through the evaluate() interface. , The symbol - indicates that the relevant metrics have not been tested yet, and the prediction time is obtained in the following environment:

- Test environment:
   - CUDA 9.0
   - CUDNN 7.5
   - PaddlePaddle v1.6
   - TensorRT-5.1.2.2
   - GPU: Tesla V100
- Test mode:
   - To make it easier to compare the reasoning speed of different models, the input is the same size as the image: 3x640x640.
   - Batch Size=1
   - Remove the first 10 rounds of warmup time and test the average time in the unit of ms/image for 100 rounds, including the time to copy the input data to the GPU, computation time, and time to copy data to CPU.
   - Use the Fluid C++ prediction engine. Enable the FP32 TensorRT configuration.
   - Start the test. FLAGS_cudnn_exhaustive_search=True: search for the convolutional algorithm using the exhaustive method.

| Model | Model size | Prediction time (ms) | BoxAP (%) | MaskAP (%) |
|:-------|:-----------|:-------------|:----------|:----------|
| [MaskRCNN-ResNet18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_r18_fpn_1x.tar) | 189.1 MB | -- | 33.6 | 30.5 |
| [MaskRCNN-ResNet50](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_2x.tar) | 143.9MB | 159.527 | 38.2 | 38.2 |
| [MaskRCNN-ResNet50-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar) | 177.7 MB | 83.567 | 38.7 | 34.7 |
| [MaskRCNN-ResNet50_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar) | 177.7 MB | 97.929 | 39.8 | 35.4 |
| [MaskRCNN-ResNet101-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar) | 253.6MB | 97.929 | 39.5 | 35.2 |
| [MaskRCNN-ResNet101_vd-FPN](https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar) | 253.7MB | 97.647 | 41.4 | 36.8 |
| [MaskRCNN - HRNet_W18-FPN](https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_hrnetv2p_w18_2x.tar) | 120.7 MB | -- | 38.7 | 34.7 |


## Semantic segmentation model

> The following metrics are tested on the MSCOCO validation set. The symbol `-` in the table indicates that the metrics have not been tested yet.

| Model | Model size | Prediction time (ms) | mIoU（%） |
|:-------|:-----------|:-------------|:----------|
| [DeepLabv3_MobileNetV2_x1.0](https://bj.bcebos.com/v1/paddleseg/deeplab_mobilenet_x1_0_coco.tgz) | 14.7MB | -- | -- |
| [DeepLabv3_Xception65](https://paddleseg.bj.bcebos.com/models/xception65_coco.tgz) | 329.3MB | -- | -- |
| [UNet](https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz) | 107.3MB | -- | -- |


> The following metrics are tested on the Cityscapes validation set. The symbol `-` in the table indicates that the metrics have not been tested yet.

| Model | Model size | Prediction time (ms) | mIoU（%） |
|:-------|:-----------|:-------------|:----------|
| [DeepLabv3_MobileNetV3_large_x1_0_ssld](https://paddleseg.bj.bcebos.com/models/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz) | 9.3 MB | -- | 73.28 |
| [DeepLabv3_MobileNetV2_x1.0](https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz) | 14.7MB | -- | 69.8 |
| [DeepLabv3_Xception65](https://paddleseg.bj.bcebos.com/models/xception65_bn_cityscapes.tgz) | 329.3MB | -- | 79.3 |
| [HRNet_W18](https://paddleseg.bj.bcebos.com/models/hrnet_w18_bn_cityscapes.tgz) | 77.3MB | -- | 79.36 |
| [Fast-SCNN](https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar) | 9.8MB | -- | 69.64 |
