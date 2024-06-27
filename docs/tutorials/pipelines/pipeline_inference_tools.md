# PaddleX 模型产线开发工具推理预测

模型产线指的是可以独立完成某类任务且具备落地能力的模型或模型组合，其可以是单模型产线，也可以是多模型组合的产线，PaddleX 提供了丰富的模型产线，可以方便地完成 AI 任务的推理和部署。云端使用请前往飞桨 [AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine)创建产线使用，本地端使用可以参考本文档， PaddleX 中的每个模型产线有多个模型可供选择，并均提供了官方权重，支持通过命令行方式直接推理预测和调用 Python API 预测。命令行方式直接推理预测可以快速体验模型推理效果，而 Python API 预测可以方便地集成到自己的项目中进行预测。

## 1.安装 PaddleX
在使用单模型开发工具之前，首先需要安装 PaddleX 的 wheel 包，安装方式请参考 [PaddleX 安装文档](../INSTALL.md)。

## 2.PaddleX 模型产线开发工具使用方式

### 2.1 通用图像分类产线
图像分类产线内置了多个图像分类的单模型，包含 `ResNet` 系列、`PP-LCNet` 系列、`MobileNetV2` 系列、`MobileNetV3` 系列、`ConvNeXt` 系列、`PP-HGNet` 系列、`PP-HGNetV2` 系列、`CLIP` 系列等模型。具体支持的分类模型列表，您可以参考[模型库](../models/support_model_list.md)，您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](../base/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。

**命令行使用方式**

您可以使用命令行将图片的类别分出来，命令行使用方式如下：

```bash
paddlex --pipeline image_classification --model PP-LCNet_x1_0 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg
```
参数解释：
- `pipeline`: 产线名称，当前支持的产线名称有 `image_classification`、`object_detection`、`semantic_segmentation`、`instance_segmentation`、`OCR`。
- `model`: 模型名称，每个产线支持的模型名称不同，请参考 [PaddleX 模型库](../models/support_model_list.md)。对于多模型组合的产线，需要指定多个模型名称，以空格分隔。
- `input`: 输入图片路径或 URL。


**Python API 使用方式**


```python
from paddlex import ClsPipeline
from paddlex import PaddleInferenceOption

model_name = "PP-LCNet_x1_0"
pipeline = ClsPipeline(model_name, kernel_option=PaddleInferenceOption())
result = pipeline.predict(
        {'input_path': "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"}
    )
print(result["cls_result"])
```  


### 2.2 通用目标检测产线


目标检测产线内置了多个目标检测的单模型，包含`RT-DETR` 系列、`PP-YOLO-E` 系列等模型。具体支持的目标检测模型列表，您可以参考[模型库](../models/support_model_list.md)，您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](../base/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。

**命令行使用方式**

您可以使用命令行将图片中的目标检测出来，命令行使用方式如下：

```bash
paddlex --pipeline object_detection --model RT-DETR-L --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png

```
参数解释：
- `pipeline`: 产线名称，当前支持的产线名称有 `image_classification`、`object_detection`、`semantic_segmentation`、`instance_segmentation`、`OCR`。
- `model`: 模型名称，每个产线支持的模型名称不同，请参考 [PaddleX 模型库](../models/support_model_list.md)。对于多模型组合的产线，需要指定多个模型名称，以空格分隔。
- `input`: 输入图片路径或 URL。

**Python API 使用方式**

```python
from pathlib import Path
from paddlex import DetPipeline
from paddlex import PaddleInferenceOption

model_name =  "RT-DETR-L"
output_base = Path("output")

output = output_base / model_name
pipeline = DetPipeline(model_name, output=output, kernel_option=PaddleInferenceOption())
result = pipeline.predict(
        {"input_path": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png"})
print(result["boxes"])

```


### 2.3 通用语义分割产线


语义分割产线内置了多个语义分割的单模型，包含`OCRNet` 系列、`DeepLabv3` 系列等模型。具体支持的语义分割模型列表，您可以参考[模型库](../models/support_model_list.md)，您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](../base/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。

**命令行使用方式**

您可以使用命令行将图片的语义信息分割出来，命令行使用方式如下：

```bash
paddlex --pipeline semantic_segmentation --model OCRNet_HRNet-W48 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png

```
参数解释：
- `pipeline`: 产线名称，当前支持的产线名称有 `image_classification`、`object_detection`、`semantic_segmentation`、`instance_segmentation`、`OCR`。
- `model`: 模型名称，每个产线支持的模型名称不同，请参考 [PaddleX 模型库](../models/support_model_list.md)。对于多模型组合的产线，需要指定多个模型名称，以空格分隔。
- `input`: 输入图片路径或 URL。

**Python API 使用方式**

```python
from pathlib import Path
from paddlex import SegPipeline
from paddlex import PaddleInferenceOption


model_name = "OCRNet_HRNet-W48",
output_base = Path("output")
output = output_base / model_name
pipeline = SegPipeline(model_name, output=output, kernel_option=PaddleInferenceOption())
result = pipeline.predict(
    {"input_path": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_semantic_segmentation_002.png"}
)
print(result["seg_map"])

```

### 2.4 通用实例分割产线


实例分割产线内置了两个目前 SOTA 的单模型，分别是 `Mask-RT-DETR-L` 和 `Mask-DT-DETR-H`。您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](../base/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。

**命令行使用方式**

您可以使用命令行将图片中的实例分割出来，命令行使用方式如下：

```bash
paddlex --pipeline instance_segmentation --model Mask-RT-DETR-L --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png

```
参数解释：
- `pipeline`: 产线名称，当前支持的产线名称有 `image_classification`、`object_detection`、`semantic_segmentation`、`instance_segmentation`、`OCR`。
- `model`: 模型名称，每个产线支持的模型名称不同，请参考 [PaddleX 模型库](../models/support_model_list.md)。对于多模型组合的产线，需要指定多个模型名称，以空格分隔。
- `input`: 输入图片路径或 URL。

**Python API 使用方式**

```python
from pathlib import Path
from paddlex import InstanceSegPipeline
from paddlex import PaddleInferenceOption

model_name =  "Mask-RT-DETR-L"
output_base = Path("output")

output = output_base / model_name
pipeline = InstanceSegPipeline(model_name, output=output, kernel_option=PaddleInferenceOption())
result = pipeline.predict(
    {"input_path": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png"})
print(result["boxes"])

```

### 2.5 OCR 产线
OCR 产线内置了 PP-OCRv4 模型，包括文本检测和文本识别两个部分。文本检测支持的模型有 `PP-OCRv4_mobile_det`、`PP-OCRv4_server_det`，文本识别支持的模型有 `PP-OCRv4_mobile_rec`、`PP-OCRv4_server_rec`。您可以使用以下两种方式进行推理预测，如果在您的场景中，上述模型不能满足您的需求，您可以参考 [PaddleX 模型训练文档](../base/README.md) 进行训练，训练后的模型可以非常方便地集成到该产线中。


**命令行使用方式**

您可以使用命令行将图片的文字识别出来，命令行使用方式如下：

```bash
paddlex --pipeline OCR --model PP-OCRv4_mobile_det PP-OCRv4_mobile_rec --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
```
参数解释：
- `pipeline`: 产线名称，当前支持的产线名称有 `image_classification`、`object_detection`、`semantic_segmentation`、`instance_segmentation`、`OCR`。
- `model`: 模型名称，每个产线支持的模型名称不同，请参考 [PaddleX 模型库](../models/support_model_list.md)。对于多模型组合的产线，需要指定多个模型名称，以空格分隔。
- `input`: 输入图片路径或 URL。
</details>

**Python API 使用方式**

```python
import cv2
from paddlex import OCRPipeline
from paddlex import PaddleInferenceOption
from paddlex.pipelines.PPOCR.utils import draw_ocr_box_txt

pipeline = OCRPipeline(
    'PP-OCRv4_mobile_det',
    'PP-OCRv4_mobile_rec',
    text_det_kernel_option=PaddleInferenceOption(),
    text_rec_kernel_option=PaddleInferenceOption(),)
result = pipeline.predict(
    {"input_path": "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"},
)

draw_img = draw_ocr_box_txt(result['original_image'],result['dt_polys'], result["rec_text"])
cv2.imwrite("ocr_result.jpg", draw_img[:, :, ::-1])
```

**注：** 更多产线推理持续更新中，敬请期待。
