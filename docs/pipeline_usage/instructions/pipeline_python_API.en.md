---
comments: true
---

# PaddleX Model Pipeline Python Usage Instructions

Before using Python scripts for rapid inference on model pipelines, please ensure you have installed PaddleX following the [PaddleX Local Installation Guide](../../installation/installation.en.md).

## I. Usage Example

Taking the image classification pipeline as an example, the usage is as follows:

```python
from paddlex import create_pipeline
pipeline = create_pipeline("image_classification")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg", batch_size=1, topk=5)
for res in output:
    res.print(json_format=False)
    res.save_to_img("./output/")
    res.save_to_json("./output/res.json")
```

In short, there are only three steps:

* Call the `create_pipeline()` method to instantiate the prediction model pipeline object;
* Call the `predict()` method of the prediction model pipeline object for inference;
* Call `print()`, `save_to_xxx()` and other related methods to print or save the prediction results.

## II. API Description

### 1. Instantiate the Prediction Model Pipeline Object by Calling `create_pipeline()`
* `create_pipeline`: Instantiates the prediction model pipeline object;
  * Parameters:
    * `pipeline`: `str` type, the pipeline name or the local pipeline configuration file path, such as "image_classification", "/path/to/image_classification.yaml";
    * `config`: `dict | None` type, pipeline configuration dictionary. If provided, `pipeline` can be omitted;
    * `device`: `str` type, used to set the inference device. If set for GPU, you can specify the card number, such as "cpu", "gpu:2". By default, using 0 id GPU if available, otherwise CPU;
    * `engine`: `str | None` type, inference engine. Available values: `paddle`, `paddle_static`, `paddle_dynamic`, `hpi`, `flexible`, `transformers`, `genai_client`;
    * `engine_config`: `dict | None` type, engine-specific configuration (flat dict for the resolved engine, or a bucketed dict keyed only by engine name; see §4.2). It can be merged and overridden per submodule;
    * `pp_option`: `PaddlePredictorOption` type, used to change inference settings (e.g. the operating mode). See "5. Compatibility Configuration (`PaddlePredictorOption`)" for details;
    * `use_hpip`：`bool | None` type, whether to enable the high-performance inference plugin (`None` for using the setting from the configuration file);
    * `hpi_config`：`dict | None` type, high-performance inference configuration;
  * Return Value: `BasePipeline` type.

### 2. Perform Inference by Calling the `predict()` Method of the Prediction Model Pipeline Object

* `predict`: Uses the defined prediction model pipeline to predict input data;
  * Parameters:
    * `input`: Any type, supporting str representing the path of the file to be predicted, or a directory containing files to be predicted, or a network URL; for CV tasks, supports numpy.ndarray representing image data; for TS tasks, supports pandas.DataFrame type data; also supports lists of the above types;
  * Return Value: `generator`, returns the prediction result of one sample per call;

### 3. Visualize the Prediction Results

The prediction results of the pipeline support to be accessed and saved, which can be achieved through corresponding attributes or methods, specifically as follows:

#### Attributes:

* `str`: `str` type representation of the prediction result;
  * Return Value: `str` type, string representation of the prediction result;
* `json`: Prediction result in JSON format;
  * Return Value: `dict` type;
* `img`: Visualization image of the prediction result;
  * Return Value: `PIL.Image` type;
* `html`: HTML representation of the prediction result;
  * Return Value: `str` type;
* _`more attrs`_: The prediction result of different pipelines support different representation methods. Please refer to the specific pipeline tutorial documentation for details.

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
* `save_to_img()`: Visualizes the prediction result and saves it as an image;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_csv()`: Saves the prediction result as a CSV file;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_html()`: Saves the prediction result as an HTML file;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* `save_to_xlsx()`: Saves the prediction result as an XLSX file;
  * Parameters:
    * `save_path`: `str` type, the path to save the result.
  * Returns: None.
* _`more funcs`_: The prediction result of different pipelines support different saving methods. Please refer to the specific pipeline tutorial documentation for details.

### 4. Inference Engine and Configuration

PaddleX pipelines support unified inference configuration via `engine` + `engine_config`, with layered control at global and submodule levels.

#### 4.1 Engine List

* `paddle`: Auto-resolved engine. When a module uses a local model directory, it is resolved to `paddle_static` or `paddle_dynamic` based on local model files; otherwise it is resolved from module support, preferring `paddle_static`;
* `paddle_static`: Paddle Inference static graph engine;
* `paddle_dynamic`: Paddle dynamic graph engine;
* `hpi`: High-performance inference plugin;
* `flexible`: Flexible runtime engine;
* `transformers`: Hugging Face Transformers-based engine;
* `genai_client`: Client engine for remote generative AI services.

#### 4.2 Flat and bucketed `engine_config`

This section describes the **shape** of the `engine_config` dict at one level (it is not a separate “configuration method” from §4.3).

At a single level (e.g. `create_pipeline(...)` or one YAML block), `engine_config` may be:

* **Flat:** a dict of options for the **resolved** engine only, e.g. `{"device_type": "gpu", "device_id": 0}` for `paddle_static`.
* **Bucketed:** a dict whose **top-level keys are only** registered engine names (`paddle_static`, `paddle_dynamic`, `hpi`, `flexible`, `transformers`, `onnxruntime`, `genai_client`), each mapping to a nested dict of options for that engine. After the final engine is chosen, only the entry for the resolved engine is used (as that engine’s flat config). Missing entry for the resolved engine yields an empty dict and a warning.

**Strict rule:** mixing bucket-style keys and flat keys at the **same** top level is **not** allowed. Use either a fully flat dict or a fully bucketed dict.

#### 4.3 Configuration Methods

**Method 1: Configure globally via `create_pipeline` arguments**

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

**Method 2: Configure in the pipeline YAML**

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

**Method 3: Global config + per-submodule override**

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

#### 4.4 Effective Rules

* `create_pipeline(..., engine=...)` has higher priority than the same field in YAML config;
* Global `engine_config` is merged with `engine_config` from submodules or sub-pipelines; fields at the lower level override global ones;
* At any level, when `engine=None`, PaddleX resolves the final engine based on the engine-selection options supported at that level; in particular, if that level supports `genai_config` and `genai_config.backend` is a server backend (such as `fastdeploy-server`, `vllm-server`, `sglang-server`, `mlx-vlm-server`, or `llama-cpp-server`), it resolves to `genai_client`;
  * Otherwise, if `use_hpip=True`, it resolves to `hpi`;
  * Otherwise, if the target model only supports `flexible`, it resolves to `flexible`;
  * Otherwise, it is equivalent to `paddle`; when a module uses a local model directory, it is resolved to `paddle_static` or `paddle_dynamic` based on local model files; otherwise it is resolved from module support, preferring `paddle_static`;
* Within the same level, `engine` has higher priority than `use_hpip` / `genai_config`;
* If a submodule or sub-pipeline does not explicitly set `engine`, but does explicitly set `use_hpip`, PaddleX re-resolves the engine from that level instead of continuing to inherit the parent `engine`;
* If a submodule does not explicitly set `engine`, but does explicitly set `genai_config.backend` to a server backend, PaddleX also re-resolves the engine from the submodule level instead of continuing to inherit the parent `engine`;
* In those cases, when that level falls back to local engine auto-resolution, it no longer inherits the parent `engine_config`; add the matching configuration at that level based on the final engine.
* When `engine` is explicitly set, `use_hpip` is ignored;
* When `engine_config` is explicitly set, `pp_option` and `hpi_config` are usually unnecessary compatibility options.

#### 4.5 Is PaddlePaddle Required?

PaddlePaddle is not required in the following scenarios:

* The relevant module runs with `engine="transformers"`;

> Note: If a module finally runs on `paddle` or `hpi`, PaddlePaddle is required. For `flexible` engine, whether PaddlePaddle is required depends on the model implementation; please refer to the corresponding model/pipeline documentation.

#### 4.6 `engine_config` Fields by Engine

The following field sets also apply to submodules in a pipeline:

* `paddle_static`:
  * `run_mode`: execution mode (`paddle`, `trt_fp32`, `trt_fp16`, `mkldnn`, etc.);
  * `device_type` / `device_id`: target device type and device index;
  * `cpu_threads`: number of CPU inference threads;
  * `delete_pass`: list of graph optimization passes to disable;
  * `enable_new_ir`: whether to enable the new IR;
  * `enable_cinn`: whether to enable CINN;
  * `trt_cfg_setting`: low-level TensorRT options passed through to backend APIs;
  * `trt_use_dynamic_shapes`: whether to enable TRT dynamic shapes;
  * `trt_collect_shape_range_info`: whether to auto-collect shape range info;
  * `trt_discard_cached_shape_range_info`: whether to drop cached shape range info and recollect;
  * `trt_dynamic_shapes`: dynamic shape map in `[min,opt,max]` format;
  * `trt_dynamic_shape_input_data`: input fill data used in dynamic-shape collection;
  * `trt_shape_range_info_path`: shape range info file path;
  * `trt_allow_rebuild_at_runtime`: whether TRT engine rebuild is allowed at runtime;
  * `mkldnn_cache_capacity`: oneDNN (MKLDNN) cache capacity.
* `paddle_dynamic`:
  * `device_type` / `device_id`: device placement for dynamic graph execution.
* `hpi`:
  * `model_name`: model name (usually auto-injected);
  * `device_type` / `device_id`: target device type and index;
  * `auto_config`: whether backend and default config are auto-selected;
  * `backend`: explicitly selected backend;
  * `backend_config`: backend-specific options;
  * `hpi_info`: model-level prior metadata (for example, dynamic shape hints);
  * `auto_paddle2onnx`: whether to auto-convert Paddle model to ONNX when needed.
* `transformers`:
  * `dtype`: model/inference dtype;
  * `device_type` / `device_id`: inference device type and device index;
  * `trust_remote_code`: whether to trust remote custom code;
  * `attn_implementation`: attention implementation;
  * `generation_config`: generation parameters;
  * `model_kwargs`: extra kwargs passed to model loading;
  * `processor_kwargs`: extra kwargs passed to processor / image processor loading;
  * `tokenizer_kwargs`: compatibility kwargs merged with `processor_kwargs`.
* `genai_client`:
  * `backend`: remote backend type;
  * `server_url`: service endpoint (`server_url` is required for server backends);
  * `max_concurrency`: max concurrent requests;
  * `client_kwargs`: extra kwargs for the client.
* `flexible`:
  * No fixed schema; fields are model-specific.

> Notes: `paddle` is an auto-resolved alias and has no dedicated `engine_config` schema. Except for `flexible`, most engines reject unknown fields.

### 5. Compatibility Configuration (`PaddlePredictorOption`)

`PaddlePredictorOption` is retained as a compatibility layer. For new projects, prefer `engine_config`.

* Effective scope: mainly compatibility settings for `paddle_static`;
* Common fields:
  * `run_mode`: execution mode (`paddle`, `trt_fp32`, `trt_fp16`, `mkldnn`, etc.);
  * `device`: inference device (for example, `cpu`, `gpu:0`);
  * `cpu_threads`: CPU inference thread count;
  * `trt_dynamic_shapes`: TensorRT dynamic shape configuration;
  * `trt_dynamic_shape_input_data`: input fill data used during dynamic-shape collection.
* Migration tip: prefer `engine + engine_config`; if both `engine_config` and `pp_option` are set, `engine_config` takes precedence.
