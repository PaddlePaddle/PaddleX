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
    * `config`：`dict | None` 类型，直接传入产线配置字典。若传入该参数，可不传 `pipeline`；
    * `device`：`str` 类型，用于设置模型推理设备，如为 GPU 则可以指定卡号，如“cpu”、“gpu:2”，默认情况下，如GPU可用，则使用GPU 0，否则使用CPU；
    * `engine`：`str | None` 类型，推理引擎。可选 `paddle`、`paddle_static`、`paddle_dynamic`、`hpi`、`flexible`、`transformers`、`genai_client`；
    * `engine_config`：`dict | None` 类型，推理引擎配置（针对解析后引擎的扁平 dict，或仅按引擎名分桶的 dict；见 4.2 节）。若设置，将传递并合并到各子模块；
    * `pp_option`：`PaddlePredictorOption` 类型，用于改变运行模式等配置项，关于推理配置的详细说明，请参考下文[5-兼容配置（PaddlePredictorOption）](#5-兼容配置paddlepredictoroption)；
    * `use_hpip`：`bool | None` 类型，是否启用高性能推理插件（`None` 表示使用配置文件中的配置）；
    * `hpi_config`：`dict | None` 类型，高性能推理配置；
  * 返回值：`BasePipeline`类型。

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

### 4. 推理引擎与配置

PaddleX 产线支持统一的 `engine` + `engine_config` 配置，并支持“全局 + 子模块”分层设置。

#### 4.1 引擎列表

* `paddle`：自动解析引擎；当模块使用本地模型目录时，根据本地模型文件解析为 `paddle_static` 或 `paddle_dynamic`；否则根据模块支持情况自动选择，优先 `paddle_static`；
* `paddle_static`：Paddle Inference 静态图推理；
* `paddle_dynamic`：Paddle 动态图推理；
* `hpi`：高性能推理插件；
* `flexible`：灵活运行时引擎；
* `transformers`：基于 Hugging Face Transformers 的推理引擎；
* `genai_client`：调用外部生成式 AI 服务的客户端引擎。

#### 4.2 扁平与分桶 `engine_config`

本节说明**同一层级**上 `engine_config` 字典的**形态**（与下一节「配置方式」并列，而不是第四种配置途径）。

在同一层级（例如 `create_pipeline(...)` 或 YAML 中某一块）中，`engine_config` 可以是：

* **扁平：** 仅针对**最终解析出的**引擎的参数字典，例如 `paddle_static` 下的 `{"device_type": "gpu", "device_id": 0}`。
* **分桶：** 顶层键**只能**为已注册的引擎名（`paddle_static`、`paddle_dynamic`、`hpi`、`flexible`、`transformers`、`onnxruntime`、`genai_client`），每个键对应一个嵌套 dict。最终引擎确定后，仅使用当前解析引擎对应键下的配置（作为该引擎的扁平配置）。若该引擎无对应键，则使用空配置并发出警告。

**严格规则：** 同一顶层**禁止**混用「分桶键」与「扁平字段」。请要么全部使用扁平 dict，要么全部使用分桶 dict。

#### 4.3 配置方式

**方式 1：通过 `create_pipeline` 传参设置全局引擎**

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="image_classification",
    device="gpu:0",
    engine="paddle_static",
    engine_config={
        "device_type": "gpu",
        "device_id": 0,
    },
)
```

**方式 2：通过产线配置文件设置引擎**

```yaml
pipeline_name: image_classification
engine: paddle_static
engine_config:
  device_type: gpu
  device_id: 0

SubModules:
  ImageClassification:
    module_name: image_classification
    model_name: PP-LCNet_x1_0
```

**方式 3：对子模块单独覆盖**

```yaml
pipeline_name: OCR
engine: paddle_static
engine_config:
  device_type: cpu
  cpu_threads: 4

SubModules:
  TextRecognition:
    module_name: text_recognition
    model_name: PP-OCRv5_server_rec
    engine: transformers
    engine_config:
      dtype: float16
      device_type: gpu
      device_id: 0
```

#### 4.4 生效规则

* `create_pipeline(..., engine=...)` 传参优先级高于配置文件中的同名字段；
* 全局 `engine_config` 会与子模块或子产线的 `engine_config` 合并，同名字段以后者优先；
* 在任一层级中，当 `engine=None` 时，会按该层支持的引擎选择参数自动解析最终引擎；其中若该层支持 `genai_config` 且 `genai_config.backend` 指向服务器后端（如 `fastdeploy-server`、`vllm-server`、`sglang-server`、`mlx-vlm-server`、`llama-cpp-server`），则解析为 `genai_client`；
  * 否则，若 `use_hpip=True`，则优先解析为 `hpi`；
  * 否则，若对应模型仅支持 `flexible`，则解析为 `flexible`；
  * 否则，等价于 `paddle`；当模块使用本地模型目录时，根据本地模型文件解析为 `paddle_static` 或 `paddle_dynamic`；否则根据模块支持情况自动选择，优先 `paddle_static`；
* 同一层级内，`engine` 的优先级高于 `use_hpip` / `genai_config`；
* 当子模块或子产线未显式设置 `engine`，但显式设置了 `use_hpip` 时，会优先按这一层重新解析引擎，而不是继续继承上一级的 `engine`；
* 当子模块未显式设置 `engine`，但显式设置了指向服务器后端的 `genai_config.backend` 时，也会优先按子模块这一层重新解析引擎，而不是继续继承上一级的 `engine`；
* 上述情况下，如果该层改为本层自动解析引擎，则不会继续继承上一级的 `engine_config`，应在该层按最终引擎补充对应配置。
* 显式设置 `engine` 时，`use_hpip` 不再生效；
* 显式设置 `engine_config` 时，`pp_option` 与 `hpi_config` 作为兼容参数通常不再需要。

#### 4.5 是否必须安装 PaddlePaddle

在以下场景中可以不安装 PaddlePaddle：

* 产线中相关模型使用 `engine="transformers"`；

> 注意：若产线中模块最终使用 `paddle` / `hpi` 引擎，仍需安装 PaddlePaddle；使用 `flexible` 引擎时，是否依赖飞桨框架取决于具体模型实现，请参考对应模型/产线文档说明。

#### 4.6 各引擎 `engine_config` 字段说明

以下字段同样适用于产线中各子模块（全局配置可被子模块覆盖）：

* `paddle_static`：
  * `run_mode`：运行模式（如 `paddle`、`trt_fp32`、`trt_fp16`、`mkldnn`）；
  * `device_type` / `device_id`：设备类型和设备编号；
  * `cpu_threads`：CPU 推理线程数；
  * `delete_pass`：手动禁用的图优化 pass 列表；
  * `enable_new_ir`：是否启用新 IR；
  * `enable_cinn`：是否启用 CINN；
  * `trt_cfg_setting`：TensorRT 底层配置透传；
  * `trt_use_dynamic_shapes`：是否启用 TRT 动态形状；
  * `trt_collect_shape_range_info`：是否自动采集 shape range 信息；
  * `trt_discard_cached_shape_range_info`：是否丢弃并重采 shape range；
  * `trt_dynamic_shapes`：动态形状配置，格式为输入名到 `[min,opt,max]`；
  * `trt_dynamic_shape_input_data`：动态形状采集时的输入填充数据；
  * `trt_shape_range_info_path`：shape range 文件路径；
  * `trt_allow_rebuild_at_runtime`：运行时是否允许重建 TRT 引擎；
  * `mkldnn_cache_capacity`：oneDNN（MKLDNN）缓存容量。
* `paddle_dynamic`：
  * `device_type` / `device_id`：动态图执行设备及编号。
* `hpi`：
  * `model_name`：模型名（通常自动注入）；
  * `device_type` / `device_id`：设备类型和编号；
  * `auto_config`：是否自动选择后端及默认配置；
  * `backend`：显式指定后端；
  * `backend_config`：后端专属配置（如 `run_mode`、TRT 精度等）；
  * `hpi_info`：模型先验信息（如动态 shape）；
  * `auto_paddle2onnx`：缺少 ONNX 模型时是否自动转换。
* `transformers`：
  * `dtype`：推理数据类型；
  * `device_type` / `device_id`：推理设备类型和设备编号；
  * `trust_remote_code`：是否信任远程自定义代码；
  * `attn_implementation`：注意力实现方式；
  * `generation_config`：文本生成参数；
  * `model_kwargs`：模型加载附加参数；
  * `processor_kwargs`：processor / image processor 加载附加参数；
  * `tokenizer_kwargs`：兼容保留的加载附加参数，会与 `processor_kwargs` 合并使用。
* `genai_client`：
  * `backend`：远端服务后端类型；
  * `server_url`：服务地址（服务器后端必填）；
  * `max_concurrency`：最大并发请求数；
  * `client_kwargs`：透传客户端参数（如鉴权参数）。
* `flexible`：
  * 无固定字段约束，按具体模型自定义解析。

> 说明：`paddle` 为自动解析引擎，不单独定义 `engine_config` 字段；除 `flexible` 外，多数引擎会校验字段并拒绝未知参数。

### 5. 兼容配置（`PaddlePredictorOption`）

`PaddlePredictorOption` 作为兼容配置保留，建议新项目优先使用 `engine_config`。

* 生效范围：主要用于 `paddle_static` 兼容配置；
* 常用字段：
  * `run_mode`：运行模式（如 `paddle`、`trt_fp32`、`trt_fp16`、`mkldnn`）；
  * `device`：推理设备（如 `cpu`、`gpu:0`）；
  * `cpu_threads`：CPU 推理线程数；
  * `trt_dynamic_shapes`：TensorRT 动态形状配置；
  * `trt_dynamic_shape_input_data`：动态形状采集时输入填充数据。
* 迁移建议：优先使用 `engine + engine_config`；当同时传入 `engine_config` 与 `pp_option` 时，以 `engine_config` 为准。
