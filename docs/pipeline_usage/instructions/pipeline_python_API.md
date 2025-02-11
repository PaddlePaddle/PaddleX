---
comments: true
---

# PaddleX模型产线Python脚本使用说明

在使用 Python 脚本进行模型产线快速推理前，请确保您已经按照 [PaddleX 本地安装教程](../../installation/installation.md)完成了 PaddleX 的安装。

## 一、使用示例

以图像分类产线为例，使用方式如下：

```python
from paddlex import create_pipeline
pipeline = create_pipeline("image_classification")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1, topk=5)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

简单来说，只需三步：

* 调用`create_pipeline()`方法实例化预测模型产线对象；
* 调用预测模型产线对象的`predict()`方法进行推理预测；
* 调用`print()`、`save_to_xxx()`等相关方法对预测结果进行打印输出或是保存。

## 二、API说明

### 1. 调用`create_pipeline()`方法实例化预测模型产线对象

* `create_pipeline`：实例化预测模型产线对象；
  * 参数：
    * `pipeline`：`str` 类型，产线名或是本地产线配置文件路径，如“image_classification”、“/path/to/image_classification.yaml”；
    * `device`：`str` 类型，用于设置模型推理设备，如为 GPU 则可以指定卡号，如“cpu”、“gpu:2”，默认情况下，如有 GPU 设置则使用 0 号 GPU，否则使用 CPU；
    * `pp_option`：`PaddlePredictorOption` 类型，用于设置模型推理后端，关于推理后端的详细说明，请参考下文[4-推理后端设置](#4-推理后端设置)；
  * 返回值：`BasePredictor`类型。

### 2. 调用预测模型产线对象的`predict()`方法进行推理预测

* `predict`：使用定义的预测模型产线，对输入数据进行预测；
  * 参数：
    * `input`：任意类型，支持str类型表示的待预测数据文件路径，或是包含待预测文件的目录，或是网络URL；对于CV任务，支持numpy.ndarray表示的图像数据；对于TS任务，支持pandas.DataFrame类型数据；同样支持上述类型所构成的list类型；
  * 返回值：`generator`，每次调用返回一个样本的预测结果；

### 3. 对预测结果进行可视化

模型产线的预测结果支持访问、可视化及保存，可通过相应的属性或方法实现，具体如下：

#### 属性：

* `str`：`str` 类型表示的预测结果；
  * 返回值：`str` 类型，预测结果的str表示；
* `json`：json格式表示的预测结果；
  * 返回值：`dict` 类型；
* `img`：预测结果的可视化图；
  * 返回值：`PIL.Image` 类型；
* `html`：预测结果的HTML表示；
  * 返回值：`str` 类型；
* _`更多`_：不同产线的预测结果支持不同的表示方式，更多属性请参考具体产线文档；

#### 方法：

* `print()`：将预测结果输出，需要注意，当预测结果不便于直接输出时，会省略相关内容；
  * 参数：
    * `json_format`：`bool`类型，默认为`False`，表示不使用json格式化输出；
    * `indent`：`int`类型，默认为`4`，当`json_format`为`True`时有效，表示json格式化的类型；
    * `ensure_ascii`：`bool`类型，默认为`False`，当`json_format`为`True`时有效；
  * 返回值：无；
* `save_to_json()`：将预测结果保存为json格式的文件，需要注意，当预测结果包含无法json序列化的数据时，会自动进行格式转换以实现序列化保存；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
    * `indent`：`int`类型，默认为`4`，当`json_format`为`True`时有效，表示json格式化的类型；
    * `ensure_ascii`：`bool`类型，默认为`False`，当`json_format`为`True`时有效；
  * 返回值：无；
* `save_to_img()`：将预测结果可视化并保存为图像；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* `save_to_csv()`：将预测结果保存为CSV文件；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* `save_to_html()`：将预测结果保存为HTML文件；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* `save_to_xlsx()`：将预测结果保存为XLSX文件；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* _`更多`_：不同产线的预测结果支持不同的存储方式，更多方法请参考具体产线文档；

### 4. 推理后端设置

PaddleX 支持通过`PaddlePredictorOption`设置推理后端，相关API如下：

#### 属性：

* `deivce`：推理设备；
  * 支持设置 `str` 类型表示的推理设备类型及卡号，设备类型支持可选 'gpu', 'cpu', 'npu', 'xpu', 'mlu'，当使用加速卡时，支持指定卡号，如使用 0 号 gpu：'gpu:0'，默认为 'gpu:0'；
  * 返回值：`str`类型，当前设置的推理设备。
* `run_mode`：推理后端；
  * 支持设置 `str` 类型的推理后端，支持可选 'paddle'，'trt_fp32'，'trt_fp16'，'trt_int8'，'mkldnn'，'mkldnn_bf16'，其中 'mkldnn' 仅当推理设备使用 cpu 时可选，默认为 'paddle'；
  * 返回值：`str`类型，当前设置的推理后端。
* `cpu_threads`：cpu 加速库计算线程数，仅当推理设备使用 cpu 时有效；
  * 支持设置 `int` 类型，cpu 推理时加速库计算线程数；
  * 返回值：`int` 类型，当前设置的加速库计算线程数。

#### 方法：

* `get_support_run_mode`：获取支持的推理后端设置；
  * 参数：无；
  * 返回值：list 类型，可选的推理后端设置。
* `get_support_device`：获取支持的运行设备类型；
  * 参数：无；
  * 返回值：list 类型，可选的设备类型。
* `get_device`：获取当前设置的设备；
  * 参数：无；
  * 返回值：str 类型。
