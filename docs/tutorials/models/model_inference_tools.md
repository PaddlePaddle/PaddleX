# PaddleX 单模型开发工具推理预测

PaddleX 提供了丰富的单模型，其是完成某一类任务的子模块的最小单元，模型开发完后，可以方便地集成到各类系统中。PaddleX 中的每个模型提供了官方权重，支持通过命令行方式直接推理预测和调用 Python API 预测。命令行方式直接推理预测可以快速体验模型推理效果，而 Python API 预测可以方便地集成到自己的项目中进行预测。

## 1.安装 PaddleX

在使用单模型开发工具之前，首先需要安装 PaddleX 的 wheel 包，安装方式请参考 [PaddleX 安装文档](../INSTALL.md)。

## 2.PaddleX 单模型开发工具使用方式

### 2.1 推理预测

PaddleX 支持单模型的统一推理 Python API，基于 Python API，您可以修改更多设置，实现多模型串联，自定义产线任务。使用 Python API 仅需几行代码，如下所示：

```python
from paddlex import PaddleInferenceOption, create_model

model_name = "PP-LCNet_x1_0"

# 实例化 PaddleInferenceOption 设置推理配置
kernel_option = PaddleInferenceOption()
kernel_option.set_device("gpu")

# 调用 create_model 函数实例化预测模型
model = create_model(model_name=model_name, kernel_option=kernel_option)

# 调用预测模型 model 的 predict 方法进行预测
result = model.predict({'input_path': "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"})
```

PaddleX 提供的所有模型均支持以上 Python API 的调用，关于模型列表，您可以参考 [PaddleX 模型列表](../models/support_model_list.md)，关于 Python API 的更多介绍，您可以参考 [PaddleX 模型推理 API](model_inference_api.md)。
