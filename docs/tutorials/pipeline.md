# PaddleX 产线（Pipeline）推理

PaddleX 中提供了多个产线，包括：OCR、图像分类、目标检测、实例分割、语义分割等，每个产线有多个模型可供选择，并均提供了官方预训练权重，支持通过 Python API、命令行方式直接推理预测。各产线使用方式可参考以下代码。

## OCR

```python
import cv2
from paddlex import OCRPipeline
from paddlex import PaddleInferenceOption
from paddle.pipelines.PPOCR.utils import draw_ocr_box_txt

pipeline = OCRPipeline(
    'PP-OCRv4_mobile_det',
    'PP-OCRv4_mobile_rec',
    text_det_kernel_option=PaddleInferenceOption(),
    text_rec_kernel_option=PaddleInferenceOption(),)
result = pipeline(
    "/paddle/dataset/paddlex/ocr_det/ocr_det_dataset_examples/images/train_img_100.jpg",
)

draw_img = draw_ocr_box_txt(result['original_image'],result['dt_polys'], result["rec_text"])
cv2.imwrite("ocr_result.jpg", draw_img[:, :, ::-1], )
```

## 图像分类

```python
from paddlex import ClsPipeline
from paddlex import PaddleInferenceOption

models = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "PP-LCNet_x0_25",
    "PP-LCNet_x0_35",
    "PP-LCNet_x0_5",
    "PP-LCNet_x0_75",
    "PP-LCNet_x1_0",
    "PP-LCNet_x1_5",
    "PP-LCNet_x2_5",
    "PP-LCNet_x2_0",
    "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5",
    "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0",
    "MobileNetV3_large_x1_25",
    "MobileNetV3_small_x0_35",
    "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75",
    "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25",
    "ConvNeXt_tiny",
    "MobileNetV2_x0_25",
    "MobileNetV2_x0_5",
    "MobileNetV2_x1_0",
    "MobileNetV2_x1_5",
    "MobileNetV2_x2_0",
    "SwinTransformer_base_patch4_window7_224",
    "PP-HGNet_small",
    "PP-HGNetV2-B0",
    "PP-HGNetV2-B4",
    "PP-HGNetV2-B6",
    "CLIP_vit_base_patch16_224",
    "CLIP_vit_large_patch14_224",
]

for model_name in models:
    try:
        pipeline = ClsPipeline(model_name, kernel_option=PaddleInferenceOption())
        result = pipeline(
            "/paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00006.jpg"
        )
        print(result["cls_result"])
    except Exception as e:
        print(f"[ERROR] model: {model_name}; err: {e}")
    print(f"[INFO] model: {model_name} done!")
```

## 目标检测

```python
from pathlib import Path
from paddlex import DetPipeline
from paddlex import PaddleInferenceOption

models = [
    "PicoDet-L",
    "PicoDet-S",
    "PP-YOLOE_plus-L",
    "PP-YOLOE_plus-M",
    "PP-YOLOE_plus-S",
    "PP-YOLOE_plus-X",
    "RT-DETR-H",
    "RT-DETR-L",
    "RT-DETR-R18",
    "RT-DETR-R50",
    "RT-DETR-X",
]
output_base = Path("output")

for model_name in models:
    output_dir = output_base / model_name
    try:
        pipeline = DetPipeline(model_name, output_dir=output_dir, kernel_option=PaddleInferenceOption())
        result = pipeline(
            "/paddle/dataset/paddlex/det/det_coco_examples/images/road0.png")
        print(result["boxes"])
    except Exception as e:
        print(f"[ERROR] model: {model_name}; err: {e}")
    print(f"[INFO] model: {model_name} done!")
```


## 实例分割

```python
from pathlib import Path
from paddlex import InstanceSegPipeline
from paddlex import PaddleInferenceOption

models = ["Mask-RT-DETR-H", "Mask-RT-DETR-L"]
output_base = Path("output")

for model_name in models:
    output_dir = output_base / model_name
    try:
        pipeline = InstanceSegPipeline(model_name, output_dir=output_dir, kernel_option=PaddleInferenceOption())
        result = pipeline(
            "/paddle/dataset/paddlex/instance_seg/instance_seg_coco_examples/images/aircraft-women-fashion-pilot-48797.png"
        )
        print(result["masks"])
    except Exception as e:
        print(f"[ERROR] model: {model_name}; err: {e}")
    print(f"[INFO] model: {model_name} done!")
```

## 语义分割


```python
from pathlib import Path
from paddlex import SegPipeline
from paddlex import PaddleInferenceOption


models = [
    "Deeplabv3-R50",
    "Deeplabv3-R101",
    "Deeplabv3_Plus-R50",
    "Deeplabv3_Plus-R101",
    "PP-LiteSeg-T",
    "OCRNet_HRNet-W48",
]

output_base = Path("output")

for model_name in models:
    output_dir = output_base / model_name
    try:
        pipeline = SegPipeline(model_name, output_dir=output_dir, kernel_option=PaddleInferenceOption())
        result = pipeline(
            "/paddle/dataset/paddlex/seg/seg_optic_examples/images/H0002.jpg"
        )
        print(result["seg_map"])
    except Exception as e:
        print(f"[ERROR] model: {model_name}; err: {e}")
    print(f"[INFO] model: {model_name} done!")
```
