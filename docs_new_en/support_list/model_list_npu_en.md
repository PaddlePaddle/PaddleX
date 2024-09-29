```markdown
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
| PP-LCNet_x```markdown
## Semantic Segmentation Module
|Model Name|mIoU (%)|Model Size (M)|
|-|-|-|
|Deeplabv3_Plus-R50 |80.36|94.9 M|
|Deeplabv3_Plus-R101|81.10|162.5 M|
|Deeplabv3-R50|79.90|138.3 M|
|Deeplabv3-R101|80.85|205.9 M|
|OCRNet_HRNet-W48|82.15|249.8 M|
|PP-LiteSeg-T|73.10|28.5 M|

**Note: The above accuracy metrics are mIoU on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset.**

## Instance Segmentation Module
|Model Name|Mask AP|Model Size (M)|
|-|-|-|
|Mask-RT-DETR-H|50.6|449.9 M|
|Mask-RT-DETR-L|45.7|113.6 M|
|Mask-RT-DETR-M|42.7|66.6 M|
|Cascade-MaskRCNN-ResNet50-FPN|36.3|254.8 M|
|Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN|39.1|254.7 M|
|PP-YOLOE_seg-S|32.5|31.5 M|

**Note: The above accuracy metrics are Mask AP(0.5:0.95) on the [COCO2017](https://cocodataset.org/#home) validation set.**

## Text Detection Module
|Model Name|Detection Hmean (%)|Model Size (M)|
|-|-|-|
|PP-OCRv4_mobile_det |77.79|4.2 M|
|PP-OCRv4_server_det |82.69|100.1 M|

**Note: The above accuracy metrics are evaluated on a self-built Chinese dataset by PaddleOCR, covering street scenes, web images, documents, and handwritten texts, with 500 images for detection.**

## Text Recognition Module
|Model Name|Recognition Avg Accuracy (%)|Model Size (M)|
|-|-|-|
|PP-OCRv4_mobile_rec |78.20|10.6 M|
|PP-OCRv4_server_rec |79.20|71.2 M|

**Note: The above accuracy metrics are evaluated on a self-built Chinese dataset by PaddleOCR, covering street scenes, web images, documents, and handwritten texts, with 11,000 images for text recognition.**

|Model Name|Recognition Avg Accuracy (%)|Model Size (M)|
|-|-|-|
|ch_SVTRv2_rec|68.81|73.9 M|

**Note: The above accuracy metrics are evaluated on the A-list of [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction).**

|Model Name|Recognition Avg Accuracy (%)|Model Size (M)|
|-|-|-|
|ch_RepSVTR_rec|65.07|22.1 M|

**Note: The above accuracy metrics are evaluated on the B-list of [PaddleOCR Algorithm Model Challenge - Task 1: OCR End-to-End Recognition](https://aistudio.baidu.com/competition/detail/1131/0/introduction).**

## Table Structure Recognition Module
|Model Name|Accuracy (%)|Model Size (M)|
|-|-|-|
|SLANet|76.31|6.9 M|

**Note: The above accuracy metrics are measured on the PubtabNet English table recognition dataset.**

## Layout Analysis Module
|Model Name|mAP (%)|Model Size (M)|
|-|-|-|
|PicoDet_layout_1x|86.8|7.4 M|

**Note: The above accuracy metrics are evaluated on a self-built layout analysis dataset by PaddleOCR, containing 10,000 images.**

## Time