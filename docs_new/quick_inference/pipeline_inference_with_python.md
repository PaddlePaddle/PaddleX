# PaddleX产线Python脚本使用说明
在使用Python脚本进行产线快速体验前，请确保您已经按照[PaddleX安装教程](/docs_new/installation/installation.md)完成了PaddleX的安装。
## 1. 使用示例
以图像分类产线为例：
```python
from pathlib import Path
from paddlex import ClsPipeline
from paddlex import PaddleInferenceOption

# 实例化 PaddleInferenceOption 设置推理配置
kernel_option = PaddleInferenceOption()
kernel_option.set_device("gpu:0")

model_name = "PP-LCNet_x1_0"
output_base = Path("output")
output = output_base / model_name

pipeline = ClsPipeline(model_name, output=output, kernel_option=kernel_option)
result = pipeline.predict(
        {'input_path': "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"}
    )
print(result["cls_result"])
```
## 2. API说明