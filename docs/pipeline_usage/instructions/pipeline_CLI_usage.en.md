---
comments: true
---

# PaddleX Pipeline CLI Usage Instructions

Before using the CLI command line for rapid inference of the pipeline, please ensure that you have completed the installation of PaddleX according to the [PaddleX Local Installation Tutorial](../../installation/installation.en.md).

## I. Usage Example

### 1. Quick Experience

Taking the image classification pipeline as an example, the usage is as follows:

```bash
paddlex --pipeline image_classification \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --device gpu:0 \
        --save_path ./output/ \
        --topk 5
```
This single step completes the inference prediction and saves the prediction results. Explanations for the relevant parameters are as follows:

* `pipeline`: The name of the pipeline or the local path to the pipeline configuration file, such as the pipeline name "image_classification", or the path to the pipeline configuration file "path/to/image_classification.yaml";
* `input`: The path to the data file to be predicted, supporting local file paths, local directories containing data files to be predicted, and file URL links;
* `engine`: Inference engine. Available values: `paddle`, `paddle_static`, `paddle_dynamic`, `hpi`, `flexible`, `transformers`, `genai_client`;
* `device`: Used to set the inference device. If set for GPU, you can specify the card number, such as "cpu", "gpu:2". By default, if a GPU is available, GPU 0 will be used; otherwise, the CPU will be used;
* `save_path`: The save path for prediction results. By default, the prediction results will not be saved;
* `use_hpip`: Enable the high-performance inference plugin;
* `hpi_config`: The high-performance inference configuration;
* _`inference hyperparameters`_: Different pipelines support different inference hyperparameter settings. And the priority of this parameter is greater than the pipeline default configuration. Such as the image classification pipeline, it supports `topk` parameter. Please refer to the specific pipeline description document for details.

### 2. Custom Pipeline Configuration

If you need to modify the pipeline, you can get the configuration file and modify it. Still taking the image classification pipeline as an example, the way to retrieve the configuration file is as follows:

```bash
paddlex --get_pipeline_config image_classification

# Please enter the path that you want to save the pipeline config file: (default `./`)
./configs/

# The pipeline config has been saved to: configs/image_classification.yaml
```

After modifying the pipeline configuration file `configs/image_classification.yaml`, such as the content for the image classification configuration file:

```yaml
pipeline_name: image_classification

SubModules:
  ImageClassification:
    module_name: image_classification
    model_name: PP-LCNet_x0_5
    model_dir: null
    batch_size: 4
    device: "gpu:0"
    topk: 5
```

Once the modification is completed, you can use this configuration file to perform model pipeline inference prediction as follows:

```bash
paddlex --pipeline configs/image_classification.yaml \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --save_path ./output/

# {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': array([0.62817, 0.03729, 0.03262, 0.03247, 0.03196]), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```

## II. PaddleX CLI Parameters (Pipeline Inference)

### 1. Common Parameters

* `--pipeline`: Pipeline name or pipeline config path (`.yaml/.yml`);
* `--input`: Input path, directory, or URL;
* `--save_path`: Directory to save results;
* `--device`: Inference device (for example, `cpu`, `gpu:0`);
* `--engine`: Set inference engine for the pipeline;
* `--use_hpip`: Enable HPIP (mainly meaningful when `--engine` is not explicitly set);
* `--hpi_config`: HPIP configuration in Python literal format (for example, `"{'backend': 'trt'}"`);
* `--get_pipeline_config`: Export default config of a pipeline.

### 2. Engine Configuration Methods

CLI currently provides `--engine` for quick engine switching. `engine_config` is configured through pipeline YAML files.

#### 2.1 Method A: Set engine directly in CLI

```bash
paddlex --pipeline image_classification \
        --input ./demo.jpg \
        --engine paddle_static \
        --device cpu
```

#### 2.2 Method B: Set `engine` and `engine_config` in YAML

```yaml
pipeline_name: image_classification
engine: paddle_static
engine_config:
  device_type: cpu
  cpu_threads: 4

SubModules:
  ImageClassification:
    module_name: image_classification
    model_name: PP-LCNet_x1_0
```

Usage:

```bash
paddlex --pipeline ./configs/image_classification.yaml \
        --input ./demo.jpg
```

#### 2.3 Method C: Global config + submodule/sub-pipeline override

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
      device_map: cuda:0
```

#### 2.4 `engine_config` Fields by Engine

For CLI usage, `engine_config` is mainly set in YAML. Common fields and meanings:

* `paddle_static`: `run_mode` (execution mode), `device_type/device_id` (target device), `cpu_threads` (CPU thread count), `delete_pass` (disabled optimization passes), `enable_new_ir`, `enable_cinn`, `trt_cfg_setting` (low-level TensorRT options), `trt_use_dynamic_shapes`, `trt_collect_shape_range_info`, `trt_discard_cached_shape_range_info`, `trt_dynamic_shapes` (`[min,opt,max]` shapes), `trt_dynamic_shape_input_data` (dynamic-shape input fill data), `trt_shape_range_info_path` (shape range file path), `trt_allow_rebuild_at_runtime` (allow TRT rebuild at runtime), `mkldnn_cache_capacity` (oneDNN cache);
* `paddle_dynamic`: `device_type/device_id` (dynamic graph execution device);
* `hpi`: `model_name` (usually auto-injected), `device_type/device_id`, `auto_config` (auto backend selection), `backend` (explicit backend), `backend_config` (backend-specific options), `hpi_info` (model prior metadata), `auto_paddle2onnx` (auto conversion to ONNX when needed);
* `transformers`: `dtype` (precision), `device_map` (placement), `trust_remote_code`, `attn_implementation`, `generation_config`, `model_kwargs`, `tokenizer_kwargs`;
* `genai_client`: `backend` (service backend type), `server_url` (service endpoint), `max_concurrency` (concurrency limit), `client_kwargs` (client passthrough options);
* `flexible`: no fixed schema.

> Note: `paddle` is an auto-resolved alias and has no dedicated `engine_config` schema. Effective values are determined by global config + submodule/sub-pipeline overrides.

### 3. Priority Rules

* `--engine` has higher priority than `engine` in YAML;
* `engine_config` is controlled by YAML (global values can be overridden in submodules or sub-pipelines);
* At any level, if `engine` is not explicitly set at that level, PaddleX resolves the final engine based on the engine-selection options supported at that level; in particular, if that level supports `genai_config` and `genai_config.backend` is a server backend, it resolves to `genai_client`; otherwise, if `use_hpip=True`, it resolves to `hpi`; otherwise, if the target model only supports `flexible`, it resolves to `flexible`; otherwise, it falls back to `paddle`, which is then auto-resolved to `paddle_static` or `paddle_dynamic` based on model files;
* Within the same level, `engine` has higher priority than `use_hpip` / `genai_config`; however, if a submodule or sub-pipeline does not explicitly set `engine` but does explicitly set `use_hpip`, PaddleX re-resolves the engine from that level instead of continuing to inherit the parent `engine`; if a submodule does not explicitly set `engine` but does explicitly set `genai_config.backend` to a server backend, PaddleX also re-resolves the engine from the submodule level;
* When a submodule or sub-pipeline falls back to local engine auto-resolution because of its own `use_hpip`, or when a submodule does so because of a server-backed `genai_config`, it no longer inherits the parent `engine_config`; add the matching config at that level based on the final engine;
* Pipeline-specific CLI args (for example, `--topk`) have higher priority than same-name fields in YAML.

### 4. Scenarios Where PaddlePaddle Is Not Required

You can run without PaddlePaddle in the following scenarios (if model and dependencies are satisfied):

* Using `transformers` engine;

> Note: If any module finally runs on `paddle` or `hpi`, PaddlePaddle is required. For `flexible` engine, whether PaddlePaddle is required depends on the model implementation; please refer to the corresponding model/pipeline documentation.
