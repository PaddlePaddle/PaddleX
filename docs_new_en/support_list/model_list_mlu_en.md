```markdown
# PaddleX Model List (Cambricon MLU)

PaddleX incorporates multiple pipelines, each containing several modules, and each module encompasses various models. You can select the appropriate models based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model size, select models with smaller storage requirements.

## Image Classification Module
| Model Name | Top-1 Accuracy (%) | Model Size (M) |
|-|-|-|
| MobileNetV3_large_x0_5 | 69.2 | 9.6 M |
| MobileNetV3_large_x0_35 | 64.3 | 7.5 M |
| MobileNetV3_large_x0_75 | 73.1 | 14.0 M |
| MobileNetV3_large_x1_0 | 75.3 | 19.5 M |
| MobileNetV3_large_x1_25 | 76.4 | 26.5 M |
| MobileNetV3_small_x0_5 | 59.2 | 6.8 M |
| MobileNetV3_small_x0_35 | 53.0 | 6.0 M |
| MobileNetV3_small_x0_75 | 66.0 | 8.5 M |
| MobileNetV3_small_x1_0 | 68.2 | 10.5 M |
| MobileNetV3_small_x1_25 | 70.7 | 13.0 M |
| PP-HGNet_small | 81.51 | 86.5 M |
| PP-LCNet_x0_5 | 63.14 | 6.7 M |
| PP-LCNet_x0_25 | 51.86 | 5.5 M |
| PP-LCNet_x0_35 | 58.09 | 5.9 M |
| PP-LCNet_x0_75 | 68.18 | 8.4 M |
| PP-LCNet_x1_0 | 71.32 | 10.5 M |
| PP-LCNet_x1_5 | 73.71 | 16.0 M |
| PP-LCNet_x2_0 | 75.18 | 23.2 M |
| PP-LCNet_x2_5 | 76.60 | 32.1 M |
| ResNet18 | 71.0 | 41.5 M |
| ResNet34 | 74.6 | 77.3 M |
| ResNet50 | 76.5 | 90.8 M |
| ResNet101 | 77.6 | 158.7 M |
| ResNet152 | 78.3 | 214.2 M |

**Note: The above accuracy metrics are Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set.**

## Object Detection Module
| Model Name | mAP (%) | Model Size (M) |
|-|-|-|
| PicoDet-L | 42.6 | 20.9 M |
| PicoDet-S | 29.1 | 4.4 M |
| PP-YOLOE_plus-L | 52.9 | 185.3 M |
| PP-YOLOE_plus-M | 49.8 | 83.2 M |
| PP-YOLOE_plus-S | 43.7 | 28.3 M |
| PP-YOLOE_plus-X | 54.7 | 349.4 M |

**Note: The above accuracy metrics are mAP(0.5:0.95) on the [COCO2017](https://cocodataset.org/#home) validation set.**

## Semantic Segmentation Module
| Model Name | mIoU (%) | Model Size (M) |
|-|-|-|
| PP-LiteSeg-T | 73.10 | 28.5 M |

**Note: The above accuracy metrics are mIoU on