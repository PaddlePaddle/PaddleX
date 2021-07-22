# PaddleX模型库

## 图像分类模型

> 表中模型准确率均为在ImageNet数据集上测试所得，表中符号`-`表示相关指标暂未测试，预测速度测试环境如下所示:

* CPU的评估是在骁龙855（SD855）上完成。
* GPU评估是在FP32+TensorRT配置下运行500次测得（去除前10次的warmup时间）。


### 移动端系列
| Model                              | Top-1 Acc | Top-5 Acc | SD855 time(ms)<br>bs=1 | Flops(G) | Params(M) | Model storage size(M) | Download Address                                                                                                      |
|----------------------------------|-----------|-----------|------------------------|----------|-----------|---------|-----------------------------------------------------------------------------------------------------------|
| MobileNetV1_<br>x0_25                | 0.5143    | 0.7546    | 3.21985                | 0.07     | 0.46      | 1.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_25_pretrained.pdparams)                |
| MobileNetV1_<br>x0_5                 | 0.6352    | 0.8473    | 9.579599               | 0.28     | 1.31      | 5.2     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_5_pretrained.pdparams)                 |
| MobileNetV1_<br>x0_75                | 0.6881    | 0.8823    | 19.436399              | 0.63     | 2.55      | 10      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_75_pretrained.pdparams)                |
| MobileNetV1                      | 0.7099    | 0.8968    | 32.523048              | 1.11     | 4.19      | 16      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_pretrained.pdparams)                      |
| MobileNetV2_<br>x0_25                | 0.5321    | 0.7652    | 3.79925                | 0.05     | 1.5       | 6.1     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_25_pretrained.pdparams)                |
| MobileNetV2_<br>x0_5                 | 0.6503    | 0.8572    | 8.7021                 | 0.17     | 1.93      | 7.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_5_pretrained.pdparams)                 |
| MobileNetV2_<br>x0_75                | 0.6983    | 0.8901    | 15.531351              | 0.35     | 2.58      | 10      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x0_75_pretrained.pdparams)                |
| MobileNetV2                      | 0.7215    | 0.9065    | 23.317699              | 0.6      | 3.44      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_pretrained.pdparams)                      |
| MobileNetV2_<br>x1_5                 | 0.7412    | 0.9167    | 45.623848              | 1.32     | 6.76      | 26      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x1_5_pretrained.pdparams)                 |
| MobileNetV2_<br>x2_0                 | 0.7523    | 0.9258    | 74.291649              | 2.32     | 11.13     | 43      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_x2_0_pretrained.pdparams)                 |
| MobileNetV3_<br>large_x1_25          | 0.7641    | 0.9295    | 28.217701              | 0.714    | 7.44      | 29      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_25_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0           | 0.7532    | 0.9231    | 19.30835               | 0.45     | 5.47      | 21      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_pretrained.pdparams)           |
| MobileNetV3_<br>large_x0_75          | 0.7314    | 0.9108    | 13.5646                | 0.296    | 3.91      | 16      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_75_pretrained.pdparams)          |
| MobileNetV3_<br>large_x0_5           | 0.6924    | 0.8852    | 7.49315                | 0.138    | 2.67      | 11      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_5_pretrained.pdparams)           |
| MobileNetV3_<br>large_x0_35          | 0.6432    | 0.8546    | 5.13695                | 0.077    | 2.1       | 8.6     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x0_35_pretrained.pdparams)          |
| MobileNetV3_<br>small_x1_25          | 0.7067    | 0.8951    | 9.2745                 | 0.195    | 3.62      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_25_pretrained.pdparams)          |
| MobileNetV3_<br>small_x1_0           | 0.6824    | 0.8806    | 6.5463                 | 0.123    | 2.94      | 12      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_pretrained.pdparams)           |
| MobileNetV3_<br>small_x0_75          | 0.6602    | 0.8633    | 5.28435                | 0.088    | 2.37      | 9.6     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_75_pretrained.pdparams)          |
| MobileNetV3_<br>small_x0_5           | 0.5921    | 0.8152    | 3.35165                | 0.043    | 1.9       | 7.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_5_pretrained.pdparams)           |
| MobileNetV3_<br>small_x0_35          | 0.5303    | 0.7637    | 2.6352                 | 0.026    | 1.66      | 6.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_35_pretrained.pdparams)          |
| MobileNetV3_<br>small_x0_35_ssld          | 0.5555    | 0.7771    | 2.6352                 | 0.026    | 1.66      | 6.9     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x0_35_ssld_pretrained.pdparams)          |
| MobileNetV3_<br>large_x1_0_ssld      | 0.7896    | 0.9448    | 19.30835               | 0.45     | 5.47      | 21      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_large_x1_0_ssld_pretrained.pdparams)      |
| MobileNetV3_small_<br>x1_0_ssld      | 0.7129    | 0.9010    | 6.5463                 | 0.123    | 2.94      | 12      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV3_small_x1_0_ssld_pretrained.pdparams)      |
| ShuffleNetV2                     | 0.6880    | 0.8845    | 10.941                 | 0.28     | 2.26      | 9       | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_0_pretrained.pdparams)                     |
| ShuffleNetV2_<br>x0_25               | 0.4990    | 0.7379    | 2.329                  | 0.03     | 0.6       | 2.7     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_25_pretrained.pdparams)               |
| ShuffleNetV2_<br>x0_33               | 0.5373    | 0.7705    | 2.64335                | 0.04     | 0.64      | 2.8     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_33_pretrained.pdparams)               |
| ShuffleNetV2_<br>x0_5                | 0.6032    | 0.8226    | 4.2613                 | 0.08     | 1.36      | 5.6     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x0_5_pretrained.pdparams)                |
| ShuffleNetV2_<br>x1_5                | 0.7163    | 0.9015    | 19.3522                | 0.58     | 3.47      | 14      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x1_5_pretrained.pdparams)                |
| ShuffleNetV2_<br>x2_0                | 0.7315    | 0.9120    | 34.770149              | 1.12     | 7.32      | 28      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_x2_0_pretrained.pdparams)                |
| ShuffleNetV2_<br>swish               | 0.7003    | 0.8917    | 16.023151              | 0.29     | 2.26      | 9.1     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ShuffleNetV2_swish_pretrained.pdparams)               |

### 其他系列

| Model                 | Top-1 Acc | Top-5 Acc | time(ms)<br>bs=1 | time(ms)<br>bs=4 | Flops(G) | Params(M) | Download Address                                                                                         |
|---------------------|-----------|-----------|-----------------------|----------------------|----------|-----------|----------------------------------------------------------------------------------------------|
| ResNet18            | 0.7098    | 0.8992    | 1.45606               | 3.56305              | 3.66     | 11.69     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_pretrained.pdparams)            |
| ResNet18_vd         | 0.7226    | 0.9080    | 1.54557               | 3.85363              | 4.14     | 11.71     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet18_vd_pretrained.pdparams)         |
| ResNet34            | 0.7457    | 0.9214    | 2.34957               | 5.89821              | 7.36     | 21.8      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_pretrained.pdparams)            |
| ResNet34_vd         | 0.7598    | 0.9298    | 2.43427               | 6.22257              | 7.39     | 21.82     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet34_vd_pretrained.pdparams)         |
| ResNet50            | 0.7650    | 0.9300    | 3.47712               | 7.84421              | 8.19     | 25.56     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_pretrained.pdparams)            |
| ResNet50_vd         | 0.7912    | 0.9444    | 3.53131               | 8.09057              | 8.67     | 25.58     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_pretrained.pdparams)         |
| ResNet101           | 0.7756    | 0.9364    | 6.07125               | 13.40573             | 15.52    | 44.55     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_pretrained.pdparams)           |
| ResNet101_vd        | 0.8017    | 0.9497    | 6.11704               | 13.76222             | 16.1     | 44.57     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_vd_pretrained.pdparams)        |
| ResNet152           | 0.7826    | 0.9396    | 8.50198               | 19.17073             | 23.05    | 60.19     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet152_pretrained.pdparams)           |
| ResNet152_vd        | 0.8059    | 0.9530    | 8.54376               | 19.52157             | 23.53    | 60.21     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet152_vd_pretrained.pdparams)        |
| ResNet200_vd        | 0.8093    | 0.9533    | 10.80619              | 25.01731             | 30.53    | 74.74     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet200_vd_pretrained.pdparams)        |
| ResNet50_vd_<br>ssld    | 0.8239    | 0.9610    | 3.53131               | 8.09057              | 8.67     | 25.58     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams)    |
| ResNet101_vd_<br>ssld   | 0.8373    | 0.9669    | 6.11704               | 13.76222             | 16.1     | 44.57     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet101_vd_ssld_pretrained.pdparams)   |
| AlexNet       | 0.567 | 0.792 | 1.44993         | 2.46696         | 1.370 | 61.090 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/AlexNet_pretrained.pdparams) |
| DarkNet53 | 0.780 | 0.941 | 4.10829 | 12.1714 | 18.580 | 41.600 | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DarkNet53_pretrained.pdparams) |
| DenseNet121 | 0.7566    | 0.9258    | 4.40447               | 9.32623              | 5.69     | 7.98      | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet121_pretrained.pdparams) |
| DenseNet161 | 0.7857    | 0.9414    | 10.39152              | 22.15555             | 15.49    | 28.68     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet161_pretrained.pdparams) |
| DenseNet169 | 0.7681    | 0.9331    | 6.43598               | 12.98832             | 6.74     | 14.15     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet169_pretrained.pdparams) |
| DenseNet201 | 0.7763    | 0.9366    | 8.20652               | 17.45838             | 8.61     | 20.01     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet201_pretrained.pdparams) |
| DenseNet264 | 0.7796    | 0.9385    | 12.14722              | 26.27707             | 11.54    | 33.37     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DenseNet264_pretrained.pdparams) |
| HRNet_W18_C | 0.7692    | 0.9339    | 7.40636          | 13.29752         | 4.14     | 21.29     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W18_C_pretrained.pdparams) |
| HRNet_W30_C | 0.7804    | 0.9402    | 9.57594          | 17.35485         | 16.23    | 37.71     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W30_C_pretrained.pdparams) |
| HRNet_W32_C | 0.7828    | 0.9424    | 9.49807          | 17.72921         | 17.86    | 41.23     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W32_C_pretrained.pdparams) |
| HRNet_W40_C | 0.7877    | 0.9447    | 12.12202         | 25.68184         | 25.41    | 57.55     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W40_C_pretrained.pdparams) |
| HRNet_W44_C | 0.7900    | 0.9451    | 13.19858         | 32.25202         | 29.79    | 67.06     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W44_C_pretrained.pdparams) |
| HRNet_W48_C | 0.7895    | 0.9442    | 13.70761         | 34.43572         | 34.58    | 77.47     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W48_C_pretrained.pdparams) |
| HRNet_W64_C | 0.7930    | 0.9461    | 17.57527         | 47.9533          | 57.83    | 128.06    | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/HRNet_W64_C_pretrained.pdparams) |
| Xception41         | 0.7930    | 0.9453    | 4.96939               | 17.01361             | 16.74    | 22.69     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception41_pretrained.pdparams)         |
| Xception65         | 0.8100    | 0.9549    | 7.26158               | 25.88778             | 25.95    | 35.48     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception65_pretrained.pdparams)         |
| Xception71         | 0.8111    | 0.9545    | 8.72457               | 31.55549             | 31.77    | 37.28     | [Download link](https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/Xception71_pretrained.pdparams)         |


## 目标检测模型

- 所有模型均在COCO17数据集中训练和测试。
- 除非特殊说明，所有ResNet骨干网络采用[ResNet-B](https://arxiv.org/pdf/1812.01187)结构。
- **推理时间(fps)**: 推理时间是在一张Tesla V100的GPU上测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。


### Faster RCNN on COCO

| 骨架网络             | 网络类型       | 推理时间(fps) | Box AP |                           下载                          |
| :------------------- | :------------- |  :------------: | :-----: | :-----------------------------------------------------: |
| ResNet50             | Faster         |            ----     |  36.7  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_1x_coco.pdparams) |
| ResNet50-vd          | Faster         |            ----     |  37.6  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_1x_coco.pdparams) |
| ResNet101            | Faster         |            ----     |  39.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_1x_coco.pdparams) |
| ResNet34-FPN         | Faster         |            ----     |  37.8  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_fpn_1x_coco.pdparams) |
| ResNet34-vd-FPN      | Faster         |            ----     |  38.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r34_vd_fpn_1x_coco.pdparams) |
| ResNet50-FPN         | Faster         |            ----     |  38.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_fpn_1x_coco.pdparams) |
| ResNet50-vd-FPN      | Faster         |            ----     |  39.5  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_1x_coco.pdparams) |
| ResNet101-vd-FPN     | Faster         |            ----     |  42.0  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r101_vd_fpn_1x_coco.pdparams) |
| ResNeXt101-vd-FPN    | Faster         |            ----     |  43.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) |
| ResNet50-vd-SSLDv2-FPN | Faster       |            ----     |  41.4  | [下载链接](https://paddledet.bj.bcebos.com/models/faster_rcnn_r50_vd_fpn_ssld_1x_coco.pdparams) |



### YOLOv3 on COCO

| 骨架网络             | 输入尺寸    |  推理时间(fps) | Box AP |                           下载                          |
| :------------------- | :------- |   :------------: | :-----: | :------------------------------------------------: |
| DarkNet53         | 608         |           ----     |  39.0  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_darknet53_270e_coco.pdparams) |
|   ResNet50_vd        | 608        |         ----     |  39.1  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r50vd_dcn_270e_coco.pdparams) |
| ResNet34         | 608         |            ----     |  36.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_r34_270e_coco.pdparams) |
| MobileNet-V1         | 608         |        ----     |  29.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_coco.pdparams) |
| MobileNet-V3         | 608         |        ----     |  31.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_coco.pdparams) |
| MobileNet-V1-SSLD    | 608         |        ----     |  31.0  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_coco.pdparams) |


### YOLOv3 on Pasacl VOC

| 骨架网络     | 输入尺寸   |推理时间(fps)| Box AP | 下载 |
| :----------- | :--: |  :-----: |:------------: |:----: |
| MobileNet-V1 | 608  |          -        |  75.2  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_270e_voc.pdparams) |
| MobileNet-V3 | 608  |            -        |  79.6  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_270e_voc.pdparams) |
| MobileNet-V1-SSLD | 608  |            -        |  78.3  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v1_ssld_270e_voc.pdparams) |
| MobileNet-V3-SSLD | 608  |            -        |  80.4  | [下载链接](https://paddledet.bj.bcebos.com/models/yolov3_mobilenet_v3_large_ssld_270e_voc.pdparams) |


### PP-YOLO on COCO

|          模型            |   骨干网络  | 输入尺寸 | Box AP<sup>val</sup> | Box AP<sup>test</sup> | V100 FP32(FPS) | V100 TensorRT FP16(FPS) | 模型下载 |
|:------------------------:|:----------:| :-------:| :------------------: | :-------------------: | :------------: | :---------------------: | :------: |
| PP-YOLO                  |     ResNet50vd |     608     |         44.8         |         45.2          |      72.9      |          155.6          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams) |
| PP-YOLOv2               |      ResNet50vd |     640     |         49.1         |         49.5          |      68.9      |          106.5          | [model](https://paddledet.bj.bcebos.com/models/ppyolov2_r50vd_dcn_365e_coco.pdparams) |
| PP-YOLOv2               |      ResNet101vd |     640     |         49.7         |         50.3          |     49.5     |         87.0         | [model](https://paddledet.bj.bcebos.com/models/ppyolov2_r101vd_dcn_365e_coco.pdparams) |

**注意:**

- PP-YOLO模型推理速度测试采用单卡V100，batch size=1进行测试，使用CUDA 10.2, CUDNN 7.5.1，TensorRT推理速度测试使用TensorRT 5.1.2.2。
- PP-YOLO模型FP32的推理速度是使用Paddle预测库进行推理速度benchmark测试结果, 且测试的均为不包含数据预处理和模型输出后处理(NMS)的数据(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。
- TensorRT FP16的速度测试相比于FP32去除了`yolo_box`(bbox解码)部分耗时，即不包含数据预处理，bbox解码和NMS(与[YOLOv4(AlexyAB)](https://github.com/AlexeyAB/darknet)测试方法一致)。


### PP-YOLO on Pascal VOC


|       模型         |   骨干网络  |   输入尺寸  | Box AP50<sup>val</sup> | 模型下载 |
|:------------------:| :----------:| :----------:| :--------------------: | :------: |
| PP-YOLO            |   ResNet50vd |     608     |          84.9          | [model](https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_voc.pdparams) |


### PP-YOLO tiny on COCO

|            模型              |   模型体积  | 后量化模型体积 |   输入尺寸  | Box AP<sup>val</sup> | Kirin 990 1xCore (FPS) | 模型下载 | 量化后模型 |
|:----------------------------:| :--------: | :------------: | :----------:| :------------------: | :--------------------: | :------: | :------: |
| PP-YOLO tiny                 |       4.2MB    |   **1.3M**     |     416     |         22.7         |          65.4         | [model](https://paddledet.bj.bcebos.com/models/ppyolo_tiny_650e_coco.pdparams) | [预测模型](https://paddledet.bj.bcebos.com/models/ppyolo_tiny_quant.tar) |

- PP-YOLO-tiny 模型推理速度测试环境配置为麒麟990芯片4线程，arm8架构。
- 我们也提供的PP-YOLO-tiny的后量化压缩模型，将模型体积压缩到**1.3M**，对精度和预测速度基本无影响



## 实例分割模型


- 所有模型均在COCO17数据集中训练和测试。
- 除非特殊说明，所有ResNet骨干网络采用[ResNet-B](https://arxiv.org/pdf/1812.01187)结构。
- **推理时间(fps)**: 推理时间是在一张Tesla V100的GPU上测试所有验证集得到，单位是fps(图片数/秒), cuDNN版本是7.5，包括数据加载、网络前向执行和后处理, batch size是1。

| 骨架网络              | 网络类型       | 推理时间(fps) | Box AP | Mask AP |                           下载                          |
| :------------------- | :------------|  :-----: | :------------: | :-----: |  :-----------------------------------------------------: |
| ResNet50             | Mask         |            ----     |  37.4  |    32.8    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_1x_coco.pdparams) |
| ResNet50             | Mask         |            ----     |  39.7  |    34.5    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_2x_coco.pdparams) |
| ResNet50-FPN         | Mask         |            ----     |  39.2  |    35.6    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_1x_coco.pdparams) |
| ResNet50-FPN         | Mask         |            ----     |  40.5  |    36.7    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_fpn_2x_coco.pdparams) |
| ResNet50-vd-FPN         | Mask         |         ----     |  40.3  |    36.4    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_1x_coco.pdparams) |
| ResNet50-vd-FPN         | Mask         |         ----     |  41.4  |    37.5    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_2x_coco.pdparams) |
| ResNet101-FPN         | Mask         |            ----     |  40.6  |    36.6    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r101_fpn_1x_coco.pdparams) |
| ResNet101-vd-FPN         | Mask         |          ----     |  42.4  |    38.1    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r101_vd_fpn_1x_coco.pdparams) |
| ResNeXt101-vd-FPN        | Mask         |          ----     |  44.0  |    39.5   | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_x101_vd_64x4d_fpn_1x_coco.pdparams) |
| ResNeXt101-vd-FPN        | Mask         |          ----     |  44.6  |    39.8   | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_x101_vd_64x4d_fpn_2x_coco.pdparams) | (https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/mask_rcnn/mask_rcnn_x101_vd_64x4d_fpn_2x_coco.yml) |
| ResNet50-vd-SSLDv2-FPN   | Mask       |            ----     |  42.0  |    38.2    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_ssld_1x_coco.pdparams) |
| ResNet50-vd-SSLDv2-FPN   | Mask       |            ----     |  42.7  |    38.9    | [下载链接](https://paddledet.bj.bcebos.com/models/mask_rcnn_r50_vd_fpn_ssld_2x_coco.pdparams) |


## 语义分割模型

> 以下指标均在Pascal VOC验证集上测试得到，表中符号`-`表示相关指标暂未测试。

| Model | Backbone | Resolution |  mIoU | Links |
|:-:|:-:|:-:|:-:|:-:|
|DeepLabV3P|ResNet50_vd|512x512|80.66%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet50_os8_voc12aug_512x512_40k/model.pdparams) |
|DeepLabV3P|ResNet101_vd|512x512|80.60%|[model](https://bj.bcebos.com/paddleseg/dygraph/pascal_voc12/deeplabv3p_resnet101_os8_voc12aug_512x512_40k/model.pdparams) |


> 以下指标均在Cityscapes验证集上测试得到，表中符号`-`表示相关指标暂未测试。

| Model | Backbone | Resolution | mIoU | Links |
|:-:|:-:|:-:|:-:|:-:|
|UNet|-|1024x512|65.00%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/unet_cityscapes_1024x512_160k/model.pdparams) |
|DeepLabV3P|ResNet50_vd|1024x512|80.36%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k/model.pdparams) |
|DeepLabV3P|ResNet101_vd|1024x512|81.10%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/deeplabv3p_resnet101_os8_cityscapes_1024x512_80k/model.pdparams) |
|Fast SCNN|-|1024x1024|69.31%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams) |
|HRNet_W18|-|1024x512|78.97%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw18_cityscapes_1024x512_80k/model.pdparams) |
|HRNet_W48|-|1024x512|80.70%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fcn_hrnetw48_cityscapes_1024x512_80k/model.pdparams) |
|BiSeNetv2|-|1024x1024|73.19%|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/bisenet_cityscapes_1024x1024_160k/model.pdparams) |
