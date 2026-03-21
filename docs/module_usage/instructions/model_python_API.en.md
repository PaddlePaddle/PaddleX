---
comments: true
---

# PaddleX Single Model Python Usage Instructions

Before using Python scripts for single model quick inference, please ensure you have completed the installation of PaddleX following the [PaddleX Local Installation Tutorial](../../installation/installation.en.md).

## I. Usage Example

Taking the image classification model as an example, the usage is as follows:

```python
from paddlex import create_model
model = create_model(model_name="PP-LCNet_x1_0")
output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```
In short, just three steps:

* Call the `create_model()` method to instantiate the prediction model object;
* Call the `predict()` method of the prediction model object to perform inference prediction;
* Call `print()`, `save_to_xxx()` and other related methods to print or save the prediction results.

## II. API Description

### 1. Instantiate the Prediction Model Object by Calling the `create_model()` Method

* `create_model`: Instantiate the prediction model object;
  * Parameters:
    * `model_name`: `str` type, model name, such as "PP-LCNet_x1_0", "/path/to/PP-LCNet_x1_0_infer/";
    * `model_dir`: `str | None` type, local path to directory of inference model files ，such as "/path/to/PP-LCNet_x1_0_infer/", default to `None`, means that use the official model specified by `model_name`;
    * `batch_size`: `int` type, default to `1`;
    * `device`: `str` type, used to set the inference device, such as "cpu", "gpu:2" for GPU settings. By default, using 0 id GPU if available, otherwise CPU;
    * `engine`: `str | None` type, inference engine. Available values: `paddle`, `paddle_static`, `paddle_dynamic`, `hpi`, `flexible`, `transformers`, `genai_client`. Default is `None`, which is auto-resolved and is typically equivalent to `paddle`;
    * `engine_config`: `dict | None` type, engine-specific configuration. See [4-Inference Engine and Configuration](#4-inference-engine-and-configuration);
    * `pp_option`: `PaddlePredictorOption` type, used to change inference settings (e.g. the operating mode). See "5. Compatibility Configuration (`PaddlePredictorOption`)" for details;
    * `use_hpip`: `bool` type, whether to enable the high-performance inference plugin (effective only when `engine=None`);
    * `hpi_config`: `dict | None` type, HPI configuration (effective when `engine="hpi"` and `engine_config` is not explicitly set);
    * `genai_config`: `dict | None` type, GenAI configuration (effective when `engine="genai_client"` and `engine_config` is not explicitly set);
    * _`inference hyperparameters`_: used to set common inference hyperparameters. Please refer to specific model description document for details.

### 2. Perform Inference Prediction by Calling the `predict()` Method of the Prediction Model Object

* `predict`: Use the defined prediction model to predict the input data;
  * Parameters:
    * `input`: Any type, supports str type representing the path of the file to be predicted, or a directory containing files to be predicted, or a network URL; for CV models, supports numpy.ndarray representing image data; for TS models, supports pandas.DataFrame type data; also supports list types composed of the above types;
  * Return Value: `generator`, using `for-in` or `next()` to iterate, and the prediction result of one sample would be returned per call.

### 3. Visualize the Prediction Results

The prediction results support to be accessed, visualized, and saved, which can be achieved through corresponding attributes or methods, specifically as follows:

#### Attributes:

* `str`: Representation of the prediction result in `str` type;
  * Returns: A `str` type, the string representation of the prediction result.
* `json`: The prediction result in JSON format;
  * Returns: A `dict` type.
* `img`: The visualization image of the prediction result. Available only when the results support visual representation;
  * Returns: A `PIL.Image` type.
* `html`: The HTML representation of the prediction result. Available only when the results support representation in HTML format;
  * Returns: A `str` type.
* _`more attrs`_: The prediction result of different models support different representation methods. Please refer to the specific model tutorial documentation for details.

#### Methods:

* `print()`: Outputs the prediction result. Note that when the prediction result is not convenient for direct output, relevant content will be omitted;
  * Parameters:
    * `json_format`: `bool` type, default is `False`, indicating that json formatting is not used;
    * `indent`: `int` type, default is `4`, valid when `json_format` is `True`, indicating the indentation level for json formatting;
    * `ensure_ascii`: `bool` type, default is `False`, valid when `json_format` is `True`;
  * Return Value: None;
* `save_to_json()`: Saves the prediction result as a JSON file. Note that when the prediction result contains data that cannot be serialized in JSON, automatic format conversion will be performed to achieve serialization and saving;
  * Parameters:
    * `save_path`: `str` type, the path to save the result;
    * `indent`: `int` type, default is `4`, valid when `json_format` is `True`, indicating the indentation level for json formatting;
    * `ensure_ascii`: `bool` type, default is `False`, valid when `json_format` is `True`;
  * Return Value: None;
* `save_to_img()`: Visualizes the prediction result and saves it as an image. Available only when the results support representation in the form of images;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_csv()`: Saves the prediction result as a CSV file. Available only when the results support representation in CSV format;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_html()`: Saves the prediction result as an HTML file. Available only when the results support representation in HTML format;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_xlsx()`: Saves the prediction result as an XLSX file. Available only when the results support representation in XLSX format;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.

### 4. Inference Engine and Configuration

PaddleX now supports unified inference configuration via `engine` + `engine_config`. This is the recommended way for new code.

#### 4.1 Engine List

* `paddle`: Auto-resolved engine. If `model_dir` is provided, it is resolved to `paddle_static` or `paddle_dynamic` based on local model files; otherwise it is resolved from model support, preferring `paddle_static`;
* `paddle_static`: Paddle Inference static graph engine;
* `paddle_dynamic`: Paddle dynamic graph engine;
* `hpi`: High-performance inference plugin;
* `flexible`: Flexible runtime engine;
* `transformers`: Hugging Face Transformers-based engine;
* `genai_client`: Client engine for remote generative AI services.

#### 4.2 Priority Rules

* When `engine=None`, PaddleX resolves the final engine in the following order:
  * If `genai_config.backend` is a server backend (such as `fastdeploy-server`, `vllm-server`, `sglang-server`, `mlx-vlm-server`, or `llama-cpp-server`), it resolves to `genai_client`;
  * Otherwise, if `use_hpip=True`, it resolves to `hpi`;
  * Otherwise, if the model only supports `flexible`, it resolves to `flexible`;
  * Otherwise, it is equivalent to `paddle`; if `model_dir` is provided, it is resolved to `paddle_static` or `paddle_dynamic` based on local model files; otherwise it is resolved from model support, preferring `paddle_static`;
* If `engine` is explicitly provided, `use_hpip` is ignored;
* If `engine_config` is explicitly provided, `pp_option`, `hpi_config`, and `genai_config` are compatibility options and will be ignored;
* Prefer using only `engine` + `engine_config` to avoid ambiguity.

#### 4.3 Is PaddlePaddle Required?

By default, most PaddleX capabilities depend on PaddlePaddle. However, PaddlePaddle is not required in these cases:

* Using `engine="transformers"` (for models that support this engine);

> Note: If your actual runtime path uses `paddle` or `hpi`, PaddlePaddle is required. For `flexible` engine, whether PaddlePaddle is required depends on the model implementation; please refer to the corresponding model/pipeline documentation.

#### 4.4 Examples

Using the Transformers engine:

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

#### 4.5 `engine_config` Fields by Engine

The following field sets are based on the current code implementation (with meanings):

* `paddle_static`:
  * `run_mode`: execution mode (for example, `paddle`, `trt_fp32`, `trt_fp16`, `mkldnn`);
  * `device_type` / `device_id`: target device type and device index;
  * `cpu_threads`: number of CPU inference threads;
  * `delete_pass`: list of graph optimization passes to disable;
  * `enable_new_ir`: whether to enable the new IR;
  * `enable_cinn`: whether to enable CINN (typically with new IR);
  * `trt_cfg_setting`: low-level TensorRT settings passed through to backend APIs;
  * `trt_use_dynamic_shapes`: whether to use TensorRT dynamic shapes;
  * `trt_collect_shape_range_info`: whether to collect shape range info automatically;
  * `trt_discard_cached_shape_range_info`: whether to discard cached shape range info and recollect it;
  * `trt_dynamic_shapes`: dynamic shape map in `[min,opt,max]` format per input;
  * `trt_dynamic_shape_input_data`: input fill data used during dynamic-shape collection;
  * `trt_shape_range_info_path`: path to the shape range info file;
  * `trt_allow_rebuild_at_runtime`: whether TensorRT engine rebuild is allowed at runtime;
  * `mkldnn_cache_capacity`: oneDNN (MKLDNN) cache capacity.
* `paddle_dynamic`:
  * `device_type` / `device_id`: device placement for dynamic graph execution.
* `hpi`:
  * `model_name`: model name (usually auto-injected internally);
  * `device_type` / `device_id`: target device and device index;
  * `auto_config`: whether backend and default config are selected automatically;
  * `backend`: explicitly selected backend;
  * `backend_config`: backend-specific options (for example, `run_mode`, TensorRT precision);
  * `hpi_info`: model-level prior metadata (for example, candidate dynamic shapes);
  * `auto_paddle2onnx`: whether to auto-convert Paddle model to ONNX when needed.
* `transformers`:
  * `dtype`: model/inference precision dtype;
  * `device_type` / `device_id`: inference device type and device index;
  * `trust_remote_code`: whether to trust and execute remote custom code from model repos;
  * `attn_implementation`: attention implementation (for example, `flash_attention_2`);
  * `generation_config`: generation parameters (for example, `max_new_tokens`, `temperature`);
  * `model_kwargs`: extra kwargs passed to model loading;
  * `processor_kwargs`: extra kwargs passed to processor / image processor loading;
  * `tokenizer_kwargs`: compatibility kwargs that are merged with `processor_kwargs`.
* `genai_client`:
  * `backend`: remote service backend type;
  * `server_url`: service endpoint (required for server backends);
  * `max_concurrency`: client-side max concurrent requests;
  * `client_kwargs`: extra kwargs passed to the OpenAI-compatible client.
* `flexible`:
  * No fixed schema; fields are model-specific.

> Notes:
> 1) `paddle` is an auto-resolved alias and does not define dedicated `engine_config` fields;
> 2) Except for `flexible`, most engines validate unknown fields strictly and raise errors for unsupported keys.

### 5. Compatibility Configuration (`PaddlePredictorOption`)

`PaddlePredictorOption` is kept for backward compatibility. For new code, prefer `engine_config`.

* Effective scope: mainly compatibility settings for `engine="paddle_static"`;
* Common fields:
  * `run_mode`: execution mode (for example, `paddle`, `trt_fp32`, `trt_fp16`, `mkldnn`);
  * `device`: inference device (for example, `cpu`, `gpu:0`);
  * `cpu_threads`: CPU inference thread count;
  * `trt_dynamic_shapes`: TensorRT dynamic shape configuration;
  * `trt_dynamic_shape_input_data`: input fill data used for dynamic-shape collection.
* Migration tip: if both `engine_config` and `pp_option` are provided, `engine_config` takes precedence. Prefer migrating to `engine + engine_config`.
