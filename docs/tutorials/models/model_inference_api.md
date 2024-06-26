# PaddleX 单模型推理 Python API 文档

PaddleX 预置了丰富的模型，并提供了 Python API 调用接口，可以方便的集成到其他项目中，或是实现多模型串联，自定义产线。

## 1. 安装 PaddleX

首先需要安装 PaddleX 的 wheel 包，安装方式请参考 [PaddleX 安装文档](../INSTALL.md)。

## 2. Python API 介绍

使用 Python API 调用模型进行预测，仅需几行代码，如下示例：

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

具体来说，需要简单几步：1. 实例化 PaddleInferenceOption 进行推理相关设置；2. 调用 create_model 实例化预测模型对象；3. 调用预测模型对象的 predict 方法进行推理预测。

#### 1. 实例化 PaddleInferenceOption 进行推理相关设置

* set_deivce：设置推理设备；
    * 参数：
        * device_setting：str 类型，推理设备类型及卡号，设备类型支持可选 'gpu', 'cpu', 'npu', 'xpu', 'mlu'，当使用加速卡时，支持指定卡号，如使用 0 号 gpu 卡：'gpu:0'，默认为 'gpu:0'；
    * 返回值：None

* set_run_mode：设置推理后端；
    * 参数：
        * run_mode：str 类型，推理后端，支持可选 'paddle'，'trt_fp32'，'trt_fp16'，'trt_int8'，'mkldnn'，'mkldnn_bf16'，其中 'mkldnn' 仅当推理设备使用 CPU 时可选，默认为 'paddle'；
    * 返回值：None

* set_cpu_threads：设置 CPU 加速库计算线程数，仅当推理设备使用 CPU 时候有效；
    * 参数：
        * cpu_threads：int 类型，CPU 推理时加速库计算线程数；
    * 返回值：None

* get_support_run_mode：获取支持的推理后端设置；
    * 参数：无；
    * 返回值：list 类型，可选的推理后端设置；

* get_support_device：获取支持的运行设备类型
    * 参数：无；
    * 返回值：list 类型，可选的设备类型；

* get_device：获取当前设置的设备；
    * 参数：无；
    * 返回值：str 类型

<!--
* set_batch_size：设置推理批大小；
    * 参数：
        * batch_size：int 类型，推理的批大小；
    * 返回值：None

* set_min_subgraph_size：设置 TensorRT 后端的最小子图大小；
    * 参数：
        * min_subgraph_size：TensorRT 后端的最小子图大小，仅当使用 trt_fp32、trt_fp16、trt_int8 后端时有效；
    * 返回值：None

* set_shape_info_filename：
    * 参数：
        * shape_info_filename：
    * 返回值：None

* set_trt_calib_mode：
    * 参数：
        * trt_calib_mode
    * 返回值：None

* set_trt_use_static：
    * 参数：
        * trt_use_static
    * 返回值：None -->

#### 2. 调用 create_model 实例化预测模型对象

* create_model：实例化预测模型对象（BasePredictor）
    * 参数：
        * model_name：str 类型，模型名
        * kernel_option：PaddleInferenceOption 类型，表示模型预测相关设置
    * 返回值：BasePredictor 类型

#### 3. 调用预测模型对象的 predict 方法进行推理预测

* predict：使用定义的预测模型，对输入数据进行预测；
    * 参数：
        * input：dict 类型，传入待预测数据，字典的 key 可通过 get_input_keys 方法获得；
    * 返回值：dict 类型，包括待预测结果和预测数据等在内的数据，如 `{'input_path': 'a/b/c.jpg', 'image': ndarray(), 'cls_pred': [0.026 0.974], 'cls_result': [{'class_ids': [2, 1]}]}`，具体内容与模型及任务相关；

* get_input_keys：
    * 参数：无
    * 返回值：list 类型，表示 predict 方法的字典参数 input 所需指定的 key，如 `['path', 'size']` 表示字典参数 input 必须包含 `'path'` 和 `'size'` 两个 key，如 `[['input_path', 'size'], ['input_data']]` 表示字典参数 input 必须包含 `'input_path'` 和 `'size'` 两个 key，**或是**包含 `'input_data'`。
