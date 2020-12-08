# PaddleX compression model library

## Image Classification

Dataset: ImageNet-1000

### Quantification

| Model | Compression strategy | Top-1 accuracy rate | Storage size | TensorRT delay (V100, ms) |
|:--:|:---:|:--:|:--:|:--:|
| MobileNetV1 | None | 70.99% | 17MB | -- |
| MobileNetV1 | Quantification | 70.18% (-0.81%) | 4.4MB | -- |
| MobileNetV2 | None | 72.15% | 15MB | -- |
| MobileNetV2 | Quantification | 71.15% (-1%) | 4.0MB | -- |
| ResNet50 | None | 76.50% | 99MB | 2.71 |
| ResNet50 | Quantification | 76.33% (-0.17%) | 25.1MB | 1.19 |

Classification model Lite delay (ms)

| Device | Model Types | Compression strategy | armv7 Thread 1 | armv7 Thread 2 | armv7 Thread 4 | armv8 Thread 1 | armv8 Thread 2 | armv8 Thread 4 |
| ------- | ----------- | ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Qualcomm 835 | MobileNetV1 | None | 96.1942 | 53.2058 | 32.4468 | 88.4955 | 47.95 | 27.5189 |
| Qualcomm 835 | MobileNetV1 | Quantification | 60.5615 | 32.4016 | 16.6596 | 56.5266 | 29.7178 | 15.1459 |
| Qualcomm 835 | MobileNetV2 | None | 65.715 | 38.1346 | 25.155 | 61.3593 | 36.2038 | 22.849 |
| Qualcomm 835 | MobileNetV2 | Quantification | 48.3495 | 30.3069 | 22.1506 | 45.8715 | 27.4105 | 18.2223 |
| Qualcomm 835 | ResNet50 | None | 526.811 | 319.6486 | 205.8345 | 506.1138 | 335.1584 | 214.8936 |
| Qualcomm 835 | ResNet50 | Quantification | 476.0507 | 256.5963 | 139.7266 | 461.9176 | 248.3795 | 149.353 |
| 高通855 | MobileNetV1 | None | 33.5086 | 19.5773 | 11.7534 | 31.3474 | 18.5382 | 10.0811 |
| 高通855 | MobileNetV1 | Quantification | 37.0498 | 21.7081 | 11.0779 | 14.0947 | 8.1926 | 4.2934 |
| 高通855 | MobileNetV2 | None | 25.0396 | 15.2862 | 9.6609 | 22.909 | 14.1797 | 8.8325 |
| 高通855 | MobileNetV2 | Quantification | 28.1631 | 18.3917 | 11.8333 | 16.9399 | 11.1772 | 7.4176 |
| 高通855 | ResNet50 | None | 185.3705 | 113.0825 | 87.0741 | 177.7367 | 110.0433 | 74.4114 |
| 高通855 | ResNet50 | Quantification | 328.2683 | 201.9937 | 106.744 | 242.6397 | 150.0338 | 79.8659 |
| Kirin 970 | MobileNetV1 | None | 101.2455 | 56.4053 | 35.6484 | 94.8985 | 51.7251 | 31.9511 |
| Kirin 970 | MobileNetV1 | Quantification | 62.4412 | 32.2585 | 16.6215 | 57.825 | 29.2573 | 15.1206 |
| Kirin 970 | MobileNetV2 | None | 70.4176 | 42.0795 | 25.1939 | 68.9597 | 39.2145 | 22.6617 |
| Kirin 970 | MobileNetV2 | Quantification | 53.0961 | 31.7987 | 21.8334 | 49.383 | 28.2358 | 18.3642 |
| Kirin 970 | ResNet50 | None | 586.8943 | 344.0858 | 228.2293 | 573.3344 | 351.4332 | 225.8006 |
| Kirin 970 | ResNet50 | Quantification | 489.6188 | 258.3279 | 142.6063 | 480.0064 | 249.5339 | 138.5284 |

### Cropping

Paddle Lite inference time consumption:

Environment: Qualcomm SnapDragon 845 + armv8

Speed index: Thread1/Thread2/Thread4 time consumption


| Model | Compression strategy | Top-1 | Storage size | Paddle Lite inference time consumption | TensorRT inference speed (FPS) |
|:--:|:---:|:--:|:--:|:--:|:--:|
| MobileNetV1 | None | 70.99% | 17MB | 66.052\35.8014\19.5762 | -- |
| MobileNetV1 | Cropping -30% | 70.4% (-0.59%) | 12MB | 46.5958\25.3098\13.6982 | -- |
| MobileNetV1 | Cropping -50% | 69.8% (-1.19%) | 9MB | 37.9892\20.7882\11.3144 | -- |

## Object Detection

### Quantification

Dataset: COCO2017

| Model | Compression strategy | Dataset | Image/GPU | Input 608 Box AP | Storage size | TensorRT delay (V100, ms) |
| :----------------------------: | :---------: | :----: | :-------: | :------------: | :------------: | :----------: |
| MobileNet-V1-YOLOv3 | None | COCO | 8 | 29.3 | 95MB | -- |
| MobileNet-V1-YOLOv3 | Quantification | COCO | 8 | 27.9 (-1.4) | 25MB | -- |
| R34-YOLOv3 | None | COCO | 8 | 36.2 | 162MB | -- |
| R34-YOLOv3 | Quantification | COCO | 8 | 35.7 (-0.5) | 42.7MB | -- |

### Cropping

Dataset: Pasacl VOC & COCO2017

Paddle Lite inference time consumption:

Environment: Qualcomm SnapDragon 845 + armv8

Speed index: Thread1/Thread2/Thread4 time consumption

| Model | Compression strategy | Dataset | Image/GPU | Input 608 Box mmAP | Storage size | Paddle Lite inference time consumption (ms)(608*608)*TensorRT inference speed (FPS) (608608) |
| :----------------------------: | :---------------: | :--------: | :-------: | :------------: | :----------: | :--------------: | :--------------: |
| MobileNet-V1-YOLOv3 | None | Pascal VOC | 8 | 76.2 | 94MB | 1238\796.943\520.101 | 60.04 |
| MobileNet-V1-YOLOv3 | Cropping -52.88% | Pascal VOC | 8 | 77.6 (+1.4) | 31MB | 602.497\353.759\222.427 | 99.36 |
| MobileNet-V1-YOLOv3 | None | COCO | 8 | 29.3 | 95MB | -- | -- |
| MobileNet-V1-YOLOv3 | Cropping-51.77% | COCO | 8 | 26.0 (-3.3) | 32MB | -- | 73.93 |

## Semantic Segmentation

Dataset: Cityscapes


### Quantification

| Model | Compression strategy | mIOU | Storage size |
| :--------------------: | :---------: | :-----------: | :------------: |
| DeepLabv3-MobileNetv2 | None | 69.81 | 7.4MB |
| DeepLabv3-MobileNetv2 | Quantification | 67.59 (-2.22) | 2.1MB |

Image segmentation model Lite delay (ms), input size 769 x 769

| Device | Model Types | Compression strategy | armv7 Thread 1 | armv7 Thread 2 | armv7 Thread 4 | armv8 Thread 1 | armv8 Thread 2 | armv8 Thread 4 |
| ------- | ---------------------- | ------------- | -------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| Qualcomm 835 | DeepLabv3-MobileNetv2 | None | 1282.8126 | 793.2064 | 653.6538 | 1193.9908 | 737.1827 | 593.4522 |
| Qualcomm 835 | DeepLabv3-MobileNetv2 | Quantification | 981.44 | 658.4969 | 538.6166 | 885.3273 | 586.1284 | 484.0018 |
| 高通855 | DeepLabv3-MobileNetv2 | None | 639.4425 | 390.1851 | 322.7014 | 477.7667 | 339.7411 | 262.2847 |
| 高通855 | DeepLabv3-MobileNetv2 | Quantification | 705.7589 | 474.4076 | 427.2951 | 394.8352 | 297.4035 | 264.6724 |
| Kirin 970 | DeepLabv3-MobileNetv2 | None | 1771.1301 | 1746.0569 | 1222.4805 | 1448.9739 | 1192.4491 | 760.606 |
| Kirin 970 | DeepLabv3-MobileNetv2 | Quantification | 1320.386 | 918.5328 | 672.2481 | 1020.753 | 820.094 | 591.4114 |

### Cropping

Paddle Lite inference time consumption:

Environment: Qualcomm SnapDragon 845 + armv8

Speed index: Thread1/Thread2/Thread4 time consumption


| Model | Compression method | mIOU | Storage size | Paddle Lite inference time consumption | TensorRT inference speed (FPS) |
| :-------: | :---------------: | :-----------: | :------: | :------------: | :----: |
| FastSCNN | None | 69.64 | 11 MB | 1226.36\682.96\415.664 | 39.53 |
| FastSCNN | Cropping -47.60% | 66.68 (-2.96) | 5.7MB | 866.693\494.467\291.748 | 51.48 |
