# PaddleX Model List (Hygon DCU)

PaddleX incorporates multiple pipelines, each containing several modules, and each module encompasses various models. The specific models to use can be selected based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model storage size, select models with smaller storage sizes.

## Image Classification Module
| Model Name | Top-1 Accuracy (%) | Model Storage Size (M) |
|-|-|-|
| ResNet18 | 71.0 | 41.5 M |
| ResNet34 | 74.6 | 77.3 M |
| ResNet50 | 76.5 | 90.8 M |
| ResNet101 | 77.6 | 158.7 M |
| ResNet152 | 78.3 | 214.2 M |

**Note: The above accuracy metrics are Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set.**

## Semantic Segmentation Module
| Model Name | mIoU (%) | Model Storage Size (M) |
|-|-|-|
| Deeplabv3_Plus-R50 | 80.36 | 94.9 M |
| Deeplabv3_Plus-R101 | 81.10 | 162.5 M |

**Note: The above accuracy metrics are mIoU on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.**
