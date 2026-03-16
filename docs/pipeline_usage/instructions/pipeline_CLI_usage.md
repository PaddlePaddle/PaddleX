---
comments: true
---

# PaddleX模型产线CLI（命令行）使用说明

在使用CLI（命令行）进行模型产线快速推理前，请确保您已经按照[PaddleX本地安装教程](../../installation/installation.md)完成了PaddleX的安装。

## 一、使用示例

### 1. 快速体验

以图像分类产线为例，使用方式如下：

```bash
paddlex --pipeline image_classification \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --device gpu:0 \
        --save_path ./output/ \
        --topk 5
```

只需一步就能完成推理预测并保存预测结果，相关参数说明如下：

* `pipeline`：模型产线名称或是模型产线配置文件的本地路径，如模型产线名 “image_classification”，或模型产线配置文件路径 “path/to/image_classification.yaml”；
* `input`：待预测数据文件路径，支持本地文件路径、包含待预测数据文件的本地目录、文件URL链接；
* `engine`：推理引擎，可选 `paddle`、`paddle_static`、`paddle_dynamic`、`hpi`、`flexible`、`transformers`、`genai_client`；
* `device`：用于设置模型推理设备，如为GPU则可以指定卡号，如 “cpu”、“gpu:2”，默认情况下，如GPU可用，则使用GPU 0，否则使用CPU；
* `save_path`：预测结果的保存路径，默认情况下，不保存预测结果；
* `use_hpip`：启用高性能推理插件；
* `hpi_config`：高性能推理配置；
* _`推理超参数`_：不同产线根据具体情况提供了不同的推理超参数设置，该参数优先级大于产线配置文件。对于图像分类产线，则支持通过 `topk` 参数设置输出的前 k 个预测结果。其他产线请参考对应的产线说明文档。

### 2. 自定义产线配置

如需对产线进行修改，可获取产线配置文件后进行修改，仍以图像分类产线为例，获取配置文件方式如下：

```bash
paddlex --get_pipeline_config image_classification

# Please enter the path that you want to save the pipeline config file: (default `./`)
./configs/

# The pipeline config has been saved to: configs/image_classification.yaml
```

然后可修改产线配置文件 `configs/image_classification.yaml`，如图像分类配置文件内容为：

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

在修改完成后，即可使用该配置文件进行模型产线推理预测，方式如下：

```bash
paddlex --pipeline configs/image_classification.yaml \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
        --save_path ./output/

# {'input_path': '/root/.paddlex/predict_input/general_image_classification_001.jpg', 'class_ids': [296, 170, 356, 258, 248], 'scores': array([0.62817, 0.03729, 0.03262, 0.03247, 0.03196]), 'label_names': ['ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'Irish wolfhound', 'weasel', 'Samoyed, Samoyede', 'Eskimo dog, husky']}
```

## 二、PaddleX CLI 参数说明（产线推理）

### 1. 通用参数

* `--pipeline`：产线名称或产线配置文件路径（`.yaml/.yml`）；
* `--input`：输入数据路径、目录或 URL；
* `--save_path`：结果保存目录；
* `--device`：推理设备（如 `cpu`、`gpu:0`）；
* `--engine`：统一指定产线推理引擎；
* `--use_hpip`：启用高性能推理插件（仅在未显式指定 `--engine` 时有意义）；
* `--hpi_config`：高性能推理插件配置，使用 Python 字面量格式传入（如 `"{'backend': 'trt'}"`）；
* `--get_pipeline_config`：导出指定产线默认配置文件。

### 2. 引擎配置方式（系统说明）

CLI 当前提供 `--engine` 参数用于快速切换推理引擎；`engine_config` 通过产线配置文件进行设置。

#### 2.1 方式 A：命令行直接指定引擎

```bash
paddlex --pipeline image_classification \
        --input ./demo.jpg \
        --engine paddle_static \
        --device cpu
```

#### 2.2 方式 B：配置文件中指定 `engine` 和 `engine_config`

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

使用方式：

```bash
paddlex --pipeline ./configs/image_classification.yaml \
        --input ./demo.jpg
```

#### 2.3 方式 C：全局配置 + 子模块/子产线覆盖

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

#### 2.4 各引擎支持的 `engine_config` 字段

CLI 下 `engine_config` 主要通过配置文件设置，常用字段及含义如下：

* `paddle_static`：`run_mode`（运行模式）、`device_type/device_id`（设备）、`cpu_threads`（CPU 线程数）、`delete_pass`（禁用 pass 列表）、`enable_new_ir`、`enable_cinn`、`trt_cfg_setting`（TRT 底层参数）、`trt_use_dynamic_shapes`、`trt_collect_shape_range_info`、`trt_discard_cached_shape_range_info`、`trt_dynamic_shapes`（`[min,opt,max]` 形状）、`trt_dynamic_shape_input_data`（动态形状填充输入）、`trt_shape_range_info_path`（shape range 文件路径）、`trt_allow_rebuild_at_runtime`（运行时重建 TRT 引擎）、`mkldnn_cache_capacity`（oneDNN 缓存）；
* `paddle_dynamic`：`device_type/device_id`（动态图执行设备）；
* `hpi`：`model_name`（一般自动注入）、`device_type/device_id`、`auto_config`（自动选后端）、`backend`（指定后端）、`backend_config`（后端参数）、`hpi_info`（模型先验信息）、`auto_paddle2onnx`（自动 Paddle2ONNX）；
* `transformers`：`dtype`（精度）、`device_map`（设备映射）、`trust_remote_code`（是否信任远程代码）、`attn_implementation`（注意力实现）、`generation_config`（生成参数）、`model_kwargs`、`tokenizer_kwargs`；
* `genai_client`：`backend`（服务后端）、`server_url`（服务地址）、`max_concurrency`（并发上限）、`client_kwargs`（客户端透传参数）；
* `flexible`：无固定字段约束。

> 说明：`paddle` 是自动解析引擎，不单独定义 `engine_config` 字段；产线实际生效时由“全局配置 + 子模块/子产线覆盖”共同决定。

### 3. 参数优先级

* `--engine` 的优先级高于产线配置文件中的 `engine`；
* `engine_config` 由配置文件控制：可在全局设置，也可在子模块或子产线中覆盖；
* 在任一层级中，当该层未显式设置 `engine` 时，会按该层支持的引擎选择参数自动解析最终引擎；其中若该层支持 `genai_config` 且 `genai_config.backend` 指向服务器后端，则解析为 `genai_client`；否则，若 `use_hpip=True`，则优先解析为 `hpi`；否则，若对应模型仅支持 `flexible`，则解析为 `flexible`；否则，回退为 `paddle`，再根据模型文件自动解析为 `paddle_static` 或 `paddle_dynamic`；
* 同一层级内，`engine` 的优先级高于 `use_hpip` / `genai_config`；但如果子模块或子产线未显式设置 `engine`，而显式设置了 `use_hpip`，则会优先按这一层重新解析引擎，而不是继续继承上一级的 `engine`；如果子模块未显式设置 `engine`，但显式设置了指向服务器后端的 `genai_config.backend`，也会优先按子模块这一层重新解析引擎；
* 当子模块或子产线因本层 `use_hpip` 改为本层自动解析引擎时，或当子模块因指向服务器后端的 `genai_config` 改为本层自动解析引擎时，不再继续继承上一级的 `engine_config`，需要在该层按最终引擎补充配置；
* 各产线专属参数（例如 `--topk`）优先级高于配置文件同名字段。

### 4. 哪些场景可以不安装 PaddlePaddle

以下场景可在未安装 PaddlePaddle 时运行（前提是模型和依赖已满足）：

* 使用 `transformers` 引擎；

> 注意：若产线中任一模块最终走 `paddle` / `hpi` 引擎，仍需安装 PaddlePaddle；使用 `flexible` 引擎时，是否依赖飞桨框架取决于具体模型实现，请参考对应模型/产线文档说明。
