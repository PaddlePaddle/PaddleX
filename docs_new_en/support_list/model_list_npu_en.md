# PaddleX Model List (Huawei Ascend NPU)

PaddleX incorporates multiple pipelines, each containing several modules, and each module encompasses various models. You can select the appropriate models based on the benchmark data below. If you prioritize model accuracy, choose models with higher accuracy. If you prioritize model size, select models with smaller storage requirements.

## Image Classification Module
| Model Name | Top-1 Accuracy (%) | Model Size (M) |
|-|-|-|
| CLIP_vit_base_patch16_224 | 85.36 | 306.5 M |
| CLIP_vit_large_patch14_224 | 88.1 | 1.04 G |
| ConvNeXt_base_224 | 83.84 | 313.9 M |
| ConvNeXt_base_384 | 84.90 | 313.9 M |
| ConvNeXt_large_224 | 84.26 | 700.7 M |
| ConvNeXt_large_384 | 85.27 | 700.7 M |
| ConvNeXt_small | 83.13 | 178.0 M |
| ConvNeXt_tiny | 82.03 | 101.4 M |
| MobileNetV1_x0_75 | 68.8 | 9.3 M |
| MobileNetV1_x1_0 | 71.0 | 15.2 M |
| MobileNetV2_x0_5 | 65.0 | 7.1 M |
| MobileNetV2_x0_25 | 53.2 | 5.5 M |
| MobileNetV2_x1_0 | 72.2 | 12.6 M |
| MobileNetV2_x1_5 | 74.1 | 25.0 M |
| MobileNetV2_x2_0 | 75.2 | 41.2 M |
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
| PP-HGNet_base | 85.0 | 249.4 M |
| PP-HGNet_small | 81.51 | 86.5 M |
| PP-HGNet_tiny | 79.83 | 52.4 M |
| PP-HGNetV2-B0 | 77.77 | 21.4 M |
| PP-HGNetV2-B1 | 79.18 | 22.6 M |
| PP-HGNetV2-B2 | 81.74 | 39.9 M |
| PP-HGNetV2-B3 | 82.98 | 57.9 M |
| PP-HGNetV2-B4 | 83.57 | 70.4 M |
| PP-HGNetV2-B5 | 84.75 | 140.8 M |
| PP-HGNetV2-B6 | 86.30 | 268.4 M |
|PP-LCNet_x0_5|63.14|6.7 M|
|PP-LCNet_x0_25|51.86|5.5 M|
|PP-LCNet_x0_35|58.09|5.9 M|
|PP-LCNet_x0_75|68.18|8.4 M|
|PP-LCNet_x1_0|71.32|10.5 M|
|PP-LCNet_x1_5|73.71|16.0 M|
|PP-LCNet_x2_0|75.18|23.2 M|
|PP-LCNet_x2_5|76.60|32.1 M|
|PP-LCNetV2_base|77.05|23.7 M|
|ResNet18_vd|72.3|41.5 M|
|ResNet18|71.0|41.5 M|
|ResNet34_vd|76.0|77.3 M|
|ResNet34|74.6|77.3 M|
|ResNet50_vd|79.1|90.8 M|
|ResNet50|76.5|90.8 M|
|ResNet101_vd|80.2|158.4 M|
|ResNet101|77.6|158.7 M|
|ResNet152_vd|80.6|214.3 M|
|ResNet152|78.3|214.2 M|
|ResNet200_vd|80.9|266.0 M|
|SwinTransformer_base_patch4_window7_224|83.37|310.5 M|
|SwinTransformer_small_patch4_window7_224|83.21|175.6 M|
|SwinTransformer_tiny_patch4_window7_224|81.10|100.1 M|

**Note: The above accuracy metrics refer to Top-1 Accuracy on the [ImageNet-1k](https://www.image-net.org/index.php) validation set.**

## Object Detection Module
| Model Name | mAP (%) | Model Size (M) |
|-|-|-|
|CenterNet-DLA-34|37.6|75.4 M|
|CenterNet-ResNet50|38.9|319.7 M|
|DETR-R50|42.3|159.3 M|
|FasterRCNN-ResNet34-FPN|37.8|137.5 M|
|FasterRCNN-ResNet50-FPN|38.4|148.1 M|
|FasterRCNN-ResNet50-vd-FPN|39.5|148.1 M|
|FasterRCNN-ResNet50-vd-SSLDv2-FPN|41.4|148.1 M|
|FasterRCNN-ResNet101-FPN|41.4|216.3 M|
|FCOS-ResNet50|39.6|124.2 M|
|PicoDet-L|42.6|20.9 M|
|PicoDet-M|37.5|16.8 M|
|PicoDet-S|29.1|4.4 M |
|PicoDet-XS|26.2|5.7M |
|PP-YOLOE_plus-L|52.9|185.3 M|
|PP-YOLOE_plus-M|49.8|83.2 M|
|PP-YOLOE_plus-S|43.7|28.3 M|
|PP-YOLOE_plus-X|54.7|349.4 M|
|RT-DETR-H|56.3|435.8 M|
|RT-DETR-L|53.0|113.7 M|
|RT-DETR-R18|46.5|70.7 M|
|RT-DETR-R50|53.1|149.1 M|
|RT-DETR-X|54.8|232.9 M|
|YOLOv3-DarkNet53|39.1|219.7 M|
|YOLOv3-MobileNetV3|31.4|83.8 M|
|YOLOv3-ResNet50_vd_DCN|40.6|163.0 M|

**Note: The above accuracy metrics are for** [COCO2017](https://cocodataset.org/#home) **validation set mAP(0.5:0.95).**

## Semantic Segmentation Module
| Model Name | mIoU (%) | Model Size (M) |
|-|-|-|
| Deeplabv3_Plus-R50 | 80.36 | 94.9 M |
| Deeplabv3_Plus-R101 | 81.10 | 162.5 M |
| Deeplabv3-R50 | 79.90 | 138.3 M |
| Deeplabv3-R101 | 80.85 | 205.9 M |
| OCRNet_HRNet-W48 | 82.15 | 249.8 M |
| PP-LiteSeg-T | 73.10 | 28.5 M |

**Note: The above accuracy metrics are for** [Cityscapes](https://www.cityscapes-dataset.com/) **dataset mIoU.**

## Instance Segmentation Module
| Model Name | Mask AP | Model Size (M) |
|-|-|-|
| Mask-RT-DETR-H | 50.6 | 449.9 M |
| Mask-RT-DETR-L | 45.7 | 113.6 M |
| Mask-RT-DETR-M | 42.7 | 66.6 M |
| Cascade-MaskRCNN-ResNet50-FPN | 36.3 | 254.8 M |
| Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN | 39.1 | 254.7 M |
| PP-YOLOE_seg-S | 32.5 | 31.5 M |

**Note: The above accuracy metrics are for** [COCO2017](https://cocodataset.org/#home) **validation set Mask AP(0.5:0.95).**

## Text Detection Module
| Model Name | Detection Hmean (%) | Model Size (M) |
|-|-|-|
| PP-OCRv4_mobile_det | 77.79 | 4.2 M |
| PP-OCRv4_server_det | 82.69 | 100.1 M |

**Note: The above accuracy metrics are evaluated on PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten scenarios, with 500 images for detection.**

## Text Recognition Module
| Model Name | Recognition Avg Accuracy (%) | Model Size (M) |
|-|-|-|
| PP-OCRv4_mobile_rec | 78.20 | 10.6 M |
| PP-OCRv4_server_rec | 79.20 | 71.2 M |

**Note: The above accuracy metrics are evaluated on PaddleOCR's self-built Chinese dataset, covering street scenes, web images, documents, and handwritten scenarios, with 11,000 images for text recognition.**

| Model Name | Recognition Avg Accuracy (%) | Model Size (M) |
|-|-|-|
| ch_SVTRv2_rec | 68.81 | 73.9 M |

**Note: The above accuracy metrics are evaluated on the [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction) A-Rank.**

| Model Name | Recognition Avg Accuracy (%) | Model Size (M) |
|-|-|-|
| ch_RepSVTR_rec | 65.07 | 22.1 M |

**Note: The above accuracy metrics are evaluated on the [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction) B-Rank.**

## Table Structure Recognition Module
| Model Name | Accuracy (%) | Model Size (M) |
|-|-|-|
| SLANet | 76.31 | 6.9 M |

**Note: The above accuracy metrics are measured on the PubtabNet English table recognition dataset.**

## Layout Analysis Module
| Model Name | mAP (%) | Model Size (M) |
|------------|---------|----------------|
| PicoDet_layout_1x | 86.8 | 7.4M |

**Note: The evaluation set for the above accuracy metrics is PaddleOCR's self-built layout analysis dataset, containing 10,000 images.**

## Time Series Forecasting Module
| Model Name | MSE | MAE | Model Size (M) |
|------------|-----|-----|----------------|
| DLinear | 0.382 | 0.394 | 72K |
| NLinear | 0.386 | 0.392 | 40K |
| Nonstationary | 0.600 | 0.515 | 55.5 M |
| PatchTST | 0.385 | 0.397 | 2.0M |
| RLinear | 0.384 | 0.392 | 40K |
| TiDE | 0.405 | 0.412 | 31.7M |
| TimesNet | 0.417 | 0.431 | 4.9M |

**Note: The above accuracy metrics are measured on the [ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar) dataset (evaluation results on the test set test.csv).**

## Time Series Anomaly Detection Module
| Model Name | Precision | Recall | F1-Score | Model Size (M) |
|------------|-----------|--------|----------|----------------|
| AutoEncoder_ad | 99.36 | 84.36 | 91.25 | 52K |
| DLinear_ad | 98.98 | 93.96 | 96.41 | 112K |
| Nonstationary_ad | 98.55 | 88.95 | 93.51 | 1.8M |
| PatchTST_ad | 98.78 | 90.70 | 94.57 | 320K |
| TimesNet_ad | 98.37 | 94.80 | 96.56 | 1.3M |

**Note: The above accuracy metrics are measured on the [PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar) dataset.**

## Time Series Classification Module
| Model Name | Acc (%) | Model Size (M) |
|------------|---------|----------------|
| TimesNet_cls | 87.5 | 792K |

**Note: The above accuracy metrics are measured on the UWaveGestureLibrary: [Training](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TRAIN.csv), [Evaluation](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv) datasets.**