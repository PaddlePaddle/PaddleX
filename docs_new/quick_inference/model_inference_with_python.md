# PaddleX单模型Python脚本使用说明
在使用Python脚本进行单模型快速体验前，请确保您已经按照[PaddleX安装教程](/docs_new/installation/installation.md)完成了PaddleX的安装。
## 1. 使用示例
以图像分类产线为例：
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
## 2. API说明