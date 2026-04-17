---
comments: true
---

# PaddleX单模型Python脚本使用说明

在使用Python脚本进行单模型快速推理前，请确保您已经按照[PaddleX本地安装教程](../../installation/installation.md)完成了PaddleX的安装。

## 一、使用示例

以图像分类模型为例，使用方式如下：

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

简单来说，只需三步：

* 调用`create_model()`方法实例化预测模型对象；
* 调用预测模型对象的`predict()`方法进行推理预测；
* 调用`print()`、`save_to_xxx()`等相关方法对预测结果进行打印输出或是保存。

## 二、API说明

### 1. 调用`create_model()`方法实例化预测模型对象

* `create_model`：实例化预测模型对象；
  * 参数：
    * `model_name`：`str` 类型，模型名，如“PP-LCNet_x1_0”；
    * `model_dir`：`str | None` 类型，本地 inference 模型文件目录路径，如“/path/to/PP-LCNet_x1_0_infer/”，默认为 `None`，表示使用`model_name`指定的官方推理模型或不使用本地模型；
    * `batch_size`：`int` 类型，默认为 `1`；
    * `device`：`str` 类型，用于设置模型推理设备，如为GPU设置则可以指定卡号，如“cpu”、“gpu:2”，默认情况下，如GPU可用，则使用GPU 0，否则使用CPU；
    * `engine`：`str | None` 类型，推理引擎，可选 `paddle`、`paddle_static`、`paddle_dynamic`、`hpi`、`flexible`、`transformers`、`genai_client`。默认为 `None`，会根据配置自动解析，常见情况下等价于 `paddle`；
    * `engine_config`：`dict | None` 类型，推理引擎配置。不同引擎支持不同字段，详见下文[4-推理引擎与配置](#4-推理引擎与配置)；
    * `pp_option`：`PaddlePredictorOption` 类型，用于改变运行模式等配置项，关于推理配置的详细说明，请参考下文[5-兼容配置（PaddlePredictorOption）](#5-兼容配置paddlepredictoroption)；
    * `use_hpip`：`bool` 类型，是否启用高性能推理插件（仅在 `engine=None` 时生效）；
    * `hpi_config`：`dict | None` 类型，高性能推理配置（当 `engine="hpi"` 且未显式传入 `engine_config` 时生效）；
    * `genai_config`：`dict | None` 类型，生成式 AI 配置（当 `engine="genai_client"` 且未显式传入 `engine_config` 时生效）；
    * _`推理超参数`_：支持常见推理超参数的修改，具体参数说明详见具体模型文档；

### 2. 调用预测模型对象的`predict()`方法进行推理预测

* `predict`：使用定义的预测模型，对输入数据进行预测；
  * 参数：
    * `input`：任意类型，支持str类型表示的待预测数据文件路径，或是包含待预测文件的目录，或是网络URL；对于CV模型，支持numpy.ndarray表示的图像数据；对于TS模型，支持pandas.DataFrame类型数据；同样支持上述类型所构成的list类型；
  * 返回值：`generator`，需通过`for-in`或`next()`方式进行遍历，每次访问返回一个样本的预测结果；

### 3. 对预测结果进行可视化

模型的预测结果支持直接访问与保存等操作，可通过相应的属性或方法实现，具体如下：

#### 属性：

* `str`：`str` 类型表示的预测结果；
  * 返回值：`str` 类型，预测结果的str表示；
* `json`：json格式表示的预测结果；
  * 返回值：`dict` 类型；
* `img`：预测结果的可视化图，仅当该模型预测结果支持可视化表示时可用；
  * 返回值：`PIL.Image` 类型；
* `html`：预测结果的HTML表示，仅当该模型预测结果支持以HTML形式表示时可用；
  * 返回值：`str` 类型；
* _`更多`_：不同模型的预测结果支持不同的表示方式，更多属性请参考具体模型文档；

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
* `save_to_img()`：将预测结果可视化并保存为图像，仅当该模型预测结果支持以图像形式表示时可用；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* `save_to_csv()`：将预测结果保存为CSV文件，仅当该模型预测结果支持以CSV形式表示时可用；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* `save_to_html()`：将预测结果保存为HTML文件，仅当该模型预测结果支持以HTML形式表示时可用；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* `save_to_xlsx()`：将预测结果保存为XLSX文件，仅当该模型预测结果支持以XLSX形式表示时可用；
  * 参数：
    * `save_path`：`str`类型，结果保存的路径；
  * 返回值：无；
* _`更多`_：不同模型的预测结果支持不同的存储方式，更多方法请参考具体模型文档；

### 4. 推理引擎与配置

PaddleX 已支持统一的 `engine` + `engine_config` 推理配置方式，推荐优先使用。

#### 4.1 引擎列表

* `paddle`：自动解析引擎；若传入 `model_dir`，则根据本地模型文件解析为 `paddle_static` 或 `paddle_dynamic`；否则根据模型支持情况自动选择，优先 `paddle_static`；
* `paddle_static`：Paddle Inference 静态图推理；
* `paddle_dynamic`：Paddle 动态图推理；
* `hpi`：高性能推理插件；
* `flexible`：灵活运行时引擎；
* `transformers`：基于 Hugging Face Transformers 的推理引擎；
* `genai_client`：调用外部生成式 AI 服务的客户端引擎。

#### 4.2 配置优先级

* 当 `engine=None` 时，会按以下顺序自动解析最终引擎：
  * 若 `genai_config.backend` 指向服务器后端（如 `fastdeploy-server`、`vllm-server`、`sglang-server`、`mlx-vlm-server`、`llama-cpp-server`），则解析为 `genai_client`；
  * 否则，若 `use_hpip=True`，则优先解析为 `hpi`；
  * 否则，若该模型仅支持 `flexible`，则解析为 `flexible`；
  * 否则，等价于 `paddle`；若传入 `model_dir`，则根据本地模型文件解析为 `paddle_static` 或 `paddle_dynamic`；否则根据模型支持情况自动选择，优先 `paddle_static`；
* 当显式传入 `engine` 时，`use_hpip` 不再生效；
* 当显式传入 `engine_config` 时，`pp_option`、`hpi_config`、`genai_config` 将作为兼容参数被忽略；
* 推荐仅使用 `engine` + `engine_config` 组合，避免混用旧参数。

#### 4.3 是否必须安装 PaddlePaddle

默认情况下，PaddleX 大多数能力依赖 PaddlePaddle；但在以下场景可不安装 PaddlePaddle：

* 使用 `engine="transformers"` 推理支持该引擎的模型；

> 注意：如果实际运行过程中涉及 `paddle` / `hpi` 等依赖本地 Paddle 能力的引擎，仍需要安装 PaddlePaddle；使用 `flexible` 引擎时，是否依赖飞桨框架取决于具体模型实现，请参考对应模型/产线文档说明。

#### 4.4 示例

使用 Transformers 引擎：

```python
from paddlex import create_model

model = create_model(
    model_name="Qwen2.5-VL-3B-Instruct",
    engine="transformers",
    engine_config={
        "dtype": "float16",
        "device_type": "gpu",
        "device_id": 0,
        "attn_implementation": "flash_attention_2",
        "processor_kwargs": {
            "use_fast": True,
        },
    },
)
```

#### 4.5 各引擎 `engine_config` 字段说明

以下字段基于当前代码中的配置模型整理（含字段含义）：

* `paddle_static`：
  * `run_mode`：运行模式（如 `paddle`、`trt_fp32`、`trt_fp16`、`mkldnn` 等）；
  * `device_type` / `device_id`：目标设备类型和设备编号；
  * `cpu_threads`：CPU 推理线程数；
  * `delete_pass`：手动禁用的图优化 pass 列表；
  * `enable_new_ir`：是否启用新 IR；
  * `enable_cinn`：是否启用 CINN（通常与新 IR 配合）；
  * `trt_cfg_setting`：TensorRT 底层配置项（按 Paddle Inference TRT 接口透传）；
  * `trt_use_dynamic_shapes`：是否启用 TRT 动态形状；
  * `trt_collect_shape_range_info`：是否自动采集 shape range 信息文件；
  * `trt_discard_cached_shape_range_info`：是否丢弃已有 shape range 并重新采集；
  * `trt_dynamic_shapes`：动态形状配置，格式为输入名到 `[min,opt,max]` 三组 shape 的映射；
  * `trt_dynamic_shape_input_data`：采集动态形状时用于填充输入张量的数据；
  * `trt_shape_range_info_path`：shape range 信息文件路径；
  * `trt_allow_rebuild_at_runtime`：运行时是否允许重建 TRT 引擎；
  * `mkldnn_cache_capacity`：oneDNN（MKLDNN）缓存容量。
* `paddle_dynamic`：
  * `device_type` / `device_id`：动态图执行时的设备类型和设备编号。
* `hpi`：
  * `model_name`：模型名（内部自动注入，一般无需手动填）；
  * `device_type` / `device_id`：推理设备类型和设备编号；
  * `auto_config`：是否由系统自动选择最优后端和默认配置；
  * `backend`：指定后端（如 `paddle` / `onnxruntime` / `tensorrt` / `openvino` / `om`）；
  * `backend_config`：后端专属配置（例如指定 `run_mode`、TRT 精度等）；
  * `hpi_info`：模型级先验信息（例如候选动态 shape）；
  * `auto_paddle2onnx`：缺少 ONNX 模型时是否自动触发 Paddle2ONNX 转换。
* `transformers`：
  * `dtype`：模型权重/推理使用的数据类型（如 `float16`）；
  * `device_type` / `device_id`：推理设备类型和设备编号；
  * `trust_remote_code`：是否信任并执行 Hugging Face 仓库中的自定义代码；
  * `attn_implementation`：注意力实现方式（如 `flash_attention_2`）；
  * `generation_config`：生成参数（如 `max_new_tokens`、`temperature` 等）；
  * `model_kwargs`：传给模型加载接口的额外参数；
  * `processor_kwargs`：传给 processor / image processor 加载接口的额外参数；
  * `tokenizer_kwargs`：兼容保留的额外加载参数，会与 `processor_kwargs` 合并使用。
* `genai_client`：
  * `backend`：远端服务类型（如 `vllm-server`、`sglang-server`）；
  * `server_url`：服务地址（服务器后端必填）；
  * `max_concurrency`：客户端最大并发请求数；
  * `client_kwargs`：透传给 OpenAI 兼容客户端的其他参数（如 `api_key`）。
* `flexible`：
  * 无固定字段约束，按具体模型自定义解析。

> 说明：
> 1) `paddle` 是自动解析引擎，不直接定义自己的 `engine_config` 字段；
> 2) 除 `flexible` 外，多数引擎对未知字段会报错，建议严格按字段名传参。

### 5. 兼容配置（`PaddlePredictorOption`）

`PaddlePredictorOption` 保留为兼容能力，建议新代码优先使用 `engine_config`。

* 生效范围：主要用于 `engine="paddle_static"` 的兼容配置；
* 常用字段：
  * `run_mode`：运行模式（如 `paddle`、`trt_fp32`、`trt_fp16`、`mkldnn`）；
  * `device`：推理设备（如 `cpu`、`gpu:0`）；
  * `cpu_threads`：CPU 推理线程数；
  * `trt_dynamic_shapes`：TensorRT 动态形状配置；
  * `trt_dynamic_shape_input_data`：动态形状采集时的输入填充数据。
* 迁移建议：当 `engine_config` 与 `pp_option` 同时传入时，优先使用 `engine_config`，建议逐步迁移到 `engine + engine_config`。
