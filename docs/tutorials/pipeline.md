# PaddleX 模型产线开发工具

PaddleX 中提供了多个模型产线，包括：OCR、图像分类、目标检测、实例分割、语义分割等，每个模型产线有多个模型可供选择，并均提供了官方权重，支持通过命令行方式直接推理预测和 Python API 预测。命令行方式直接推理预测可以快速体验模型推理效果，而 Python API 预测可以方便地集成到自己的项目中进行预测。

## 1.安装 PaddleX
在使用模型产线工具之前，首先需要安装 PaddleX，安装方式请参考 [PaddleX 安装文档](xxx)。


## 2.PaddleX 模型产线工具使用方式
### 2.1 OCR 产线
OCR 产线内置了 PP-OCRv4 模型，包括文字检测和文字识别两个部分。文字检测支持的模型有`PP-OCRv4_mobile_det`、`PP-OCRv4_server_det`，文字识别支持的模型有`PP-OCRv4_mobile_rec`、`PP-OCRv4_server_rec`。您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](./train/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。

<details>
<summary><b> 命令行使用方式 </b></summary>
您可以使用命令行将图片的文字识别出来，命令行使用方式如下：

```
paddlex --task ocrdet --model PP-OCRv4_mobile_det --image /paddle/dataset/paddlex/ocr_det/ocr_det_dataset/xxx
```
参数解释：
- `task`: 任务类型，当前支持 `ocrdet`
- `model`: 模型名称，当前支持 `PP-OCRv4_mobile_det` 和 `PP-OCRv4_mobile_rec`。
</details>

<details>
<summary><b> Python API 使用方式</b></summary>

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

参数解释：
- `task`: 任务类型，当前支持 `ocrdet`
- `model`: 模型名称，当前支持 `PP-OCRv4_mobile_det` 和 `PP-OCRv4_mobile_rec`。
</details>


## 2.2 图像分类产线
图像分类产线内置了多个图像分类的单模型，包含 `ResNet` 系列、`PP-LCNet` 系列、`MobileNetV2` 系列、`MobileNetV3` 系列、`ConvNeXt` 系列、`SwinTransformer` 系列、`PP-HGNet` 系列、`PP-HGNetV2` 系列、`CLIP` 系列等模型。具体支持的分类模型列表，您可以参考[模型库](./models/support_model_list.md)，您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](./train/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。

<details>
<summary><b> 命令行使用方式 </b></summary>
您可以使用命令行将图片的文字识别出来，命令行使用方式如下：

```
paddlex --task ocrdet --model PP-OCRv4_mobile_det --image /paddle/dataset/paddlex/ocr_det/ocr_det_dataset/xxx
```
参数解释：
- `task`: 任务类型，当前支持 `ocrdet`
- `model`: 模型名称，当前支持 `PP-OCRv4_mobile_det` 和 `PP-OCRv4_mobile_rec`。
</details>


<details>
<summary><b> Python API 使用方式</b></summary>

```python
from paddlex import ClsPipeline
from paddlex import PaddleInferenceOption

pipeline = ClsPipeline(model_name, kernel_option=PaddleInferenceOption())
    result = pipeline(
        "/paddle/dataset/paddlex/cls/cls_flowers_examples/images/image_00006.jpg"
    )
    print(result["cls_result"])

</details>


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
