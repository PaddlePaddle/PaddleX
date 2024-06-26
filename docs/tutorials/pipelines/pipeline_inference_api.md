# PaddleX 模型产线 Python API 文档

PaddleX 提供了多个实用的模型产线，模型产线由一个或多个模型组合而成，面临，能够直接落地应用

## 1. 安装 PaddleX

首先需要安装 PaddleX 的 wheel 包，安装方式请参考 [PaddleX 安装文档](../INSTALL.md)。

## 2. Python API 介绍

使用 Python API 调用模型产线进行预测，仅需几行代码，如下示例：

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

如上代码所示，具体来说需要简单几步：1. 实例化 PaddleInferenceOption 进行推理相关设置；2. 实例化模型产线对象；3. 调用模型产线对象的 predict 方法进行推理预测。

#### 1. 实例化 PaddleInferenceOption 进行推理相关设置


* set_deivce：设置推理设备；
    * 参数：
        * device_setting：str 类型，推理设备类型及卡号，设备类型支持可选 'gpu', 'cpu', 'npu', 'xpu', 'mlu'，当使用加速卡时，支持指定卡号，如使用 0 号 gpu 卡：'gpu:0'，默认为 'gpu:0'；
    * 返回值：None；

* set_run_mode：设置推理后端；
    * 参数：
        * run_mode：str 类型，推理后端，支持可选 'paddle'，'trt_fp32'，'trt_fp16'，'trt_int8'，'mkldnn'，'mkldnn_bf16'，其中 'mkldnn' 仅当推理设备使用 CPU 时可选，默认为 'paddle'；
    * 返回值：None；

* set_cpu_threads：设置 CPU 加速库计算线程数，仅当推理设备使用 CPU 时候有效；
    * 参数：
        * cpu_threads：int 类型，CPU 推理时加速库计算线程数；
    * 返回值：None；

* get_support_run_mode：获取支持的推理后端设置；
    * 参数：无；
    * 返回值：list 类型，可选的推理后端设置；

* get_support_device：获取支持的运行设备类型
    * 参数：无；
    * 返回值：list 类型，可选的设备类型；

* get_device：获取当前设置的设备；
    * 参数：无；
    * 返回值：str 类型；

#### 2. 实例化模型产线对象

从 paddlex 导入对应的产线类并实例化，如图像分类产线 `ClsPipeline`，更多模型产线查看[模型产线](support_pipeline_list.md)。

#### 3. 调用预测模型对象的 predict 方法进行推理预测

* predict：使用定义的预测模型，对输入数据进行预测；
    * 参数：
        * input：dict 类型，传入待预测数据，字典的 key 可通过 get_input_keys 方法获得；
    * 返回值：dict 类型，包括待预测结果和预测数据等在内的数据，如 `{'input_path': 'a/b/c.jpg', 'image': ndarray(), 'cls_pred': [0.026 0.974], 'cls_result': [{'class_ids': [2, 1]}]}`，具体内容与模型及任务相关；

* get_input_keys：
    * 参数：无
    * 返回值：list 类型，表示 predict 方法的字典参数 input 所需指定的 key，如 `['path', 'size']` 表示字典参数 input 必须包含 `'path'` 和 `'size'` 两个 key，如 `[['input_path', 'size'], ['input_data']]` 表示字典参数 input 必须包含 `'input_path'` 和 `'size'` 两个 key，**或是**包含 `'input_data'`。
