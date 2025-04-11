---
comments: true
---

# PaddleX 高性能推理指南

在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速。本文档将首先介绍高性能推理插件的安装和使用方式，然后列举目前支持使用高性能推理插件的产线与模型。

## 目录

- [1. 基础使用方法](#1.-基础使用方法)
  - [1.1 安装高性能推理插件](#1.1-安装高性能推理插件)
  - [1.2 启用高性能推理插件](#1.2-启用高性能推理插件)
- [2. 进阶使用方法](#2.-进阶使用方法)
  - [2.1 修改高性能推理配置](#2.1-修改高性能推理配置)
  - [2.2 自定义编译高性能推理插件](#2.2-自定义编译高性能推理插件)
- [3. 常见问题](#3.-常见问题)

## 1. 基础使用方法

使用高性能推理插件前，请确保您已经按照[PaddleX本地安装教程](../installation/installation.md) 完成了PaddleX的安装，且按照PaddleX产线命令行使用说明或PaddleX产线Python脚本使用说明跑通了产线的快速推理。

### 1.1 安装高性能推理插件

* 注意：若您使用的是 Windows 系统，请参考[PaddleX本地安装教程——2.1基于Docker获取PaddleX](../installation/installation.md#21-基于docker获取paddlex) 使用 Docker 启动 PaddleX 容器。启动容器后，您可以继续阅读本指南以使用高性能推理。

根据设备类型，执行如下指令，安装高性能推理插件：

如果你的设备是 CPU，请使用以下命令安装 PaddleX 的 CPU 版本：

```bash
paddlex --install hpi-cpu
```

如果你的设备是 GPU，请使用以下命令安装 PaddleX 的 GPU 版本。请注意，GPU 版本包含了 CPU 版本的所有功能，因此无需单独安装 CPU 版本：

```bash
paddlex --install hpi-gpu
```

目前高性能推理支持的处理器架构、操作系统、设备类型和 Python 版本如下表所示：

<table>
  <tr>
    <th>处理器架构</th>
    <th>操作系统</th>
    <th>设备类型</th>
    <th>Python 版本</th>
  </tr>
  <tr>
    <td rowspan="4">x86-64</td>
    <td rowspan="4">Linux</td>
  </tr>
  <tr>
    <td>CPU</td>
    <td>3.8–3.12</td>
  </tr>
  <tr>
    <td>GPU&nbsp;（CUDA&nbsp;11.8&nbsp;+&nbsp;cuDNN&nbsp;8.6）</td>
    <td>3.8–3.12</td>
  </tr>
</table>

### 1.2 启用高性能推理插件

对于 PaddleX CLI，指定 `--use_hpip`，即可启用高性能推理插件。以通用图像分类产线和图像分类模块为例：

通用图像分类产线：

```bash
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip
```

图像分类模块：

```bash
python main.py \
    -c paddlex/configs/modules/image_classification/ResNet18.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.input=https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    -o Global.device=gpu:0 \
    -o Predict.use_hpip=True
```

对于 PaddleX Python API，启用高性能推理插件的方法类似。以通用图像分类产线和图像分类模块为例：

通用图像分类产线：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="image_classification",
    device="gpu",
    use_hpip=True
)

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

图像分类模块：

```python
from paddlex import create_model

model = create_model(
    model_name="ResNet18",
    device="gpu",
    use_hpip=True
)

output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

启用高性能推理插件得到的推理结果与未启用插件时一致。对于部分模型，在首次启用高性能推理插件时，可能需要花费较长时间完成推理引擎的构建。PaddleX 将在推理引擎的第一次构建完成后将相关信息缓存在模型目录，并在后续复用缓存中的内容以提升初始化速度。

## 2. 进阶使用方法

### 2.1 修改高性能推理配置

高性能推理配置默认使用产线配置文件，可以通过修改产线配置文件、传递CLI或Python API参数中的 `hpi_config` 字段内容来修改配置。传递CLI或Python API参数将覆盖产线配置文件的设置。

常用高性能推理配置包含以下字段：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>auto_config</code></td>
<td>是否启用自动配置模式</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
  <td><code>backend</code></td>
  <td>如果非None，可以用于指定要使用的推理后端。在手动配置模式下，不能为None。</td>
  <td><code>str | None</code></td>
  <td><code>None</code></td>
</tr>
<tr>
  <td><code>backend_config</code></td>
  <td>如果非None，则可以覆盖推理后端的默认配置项。</td>
  <td><code>dict | None</code></td>
  <td><code>None</code></td>
</tr>
<tr>
  <td><code>auto_paddle2onnx</code></td>
  <td>是否启用模型格式自动转换功能。高性能推理插件自动将模型转换为 ONNX 格式后用推理引擎推理。在需要的时候（例如用户指定了必须使用ONNX格式模型的推理后端，或者paddlex经过分析认为使用这样的后端能取得更好的推理性能）将Paddle格式模型自动转换为ONNX格式模型，转换得到的ONNX格式模型存储在原始模型目录中。此功能仅在安装了Paddle2ONNX插件时生效。</td>
  <td><code>bool</code></td>
  <td><code>True</code></td>
</tr>
</tbody>
</table>

`backend` 可选值如下表所示：

<table>
  <tr>
    <th>选项</th>
    <th>描述</th>
    <th>支持设备</th>
  </tr>
  <tr>
    <td><code>openvino</code></td>
    <td><a href="https://github.com/openvinotoolkit/openvino">OpenVINO</a>，Intel 提供的深度学习推理工具，优化了多种 Intel 硬件上的模型推理性能。</td>
    <td>CPU</td>
  </tr>
  <tr>
    <td><code>onnxruntime</code></td>
    <td><a href="https://onnxruntime.ai/">ONNX Runtime</a>，跨平台、高性能的推理引擎。</td>
    <td>CPU, GPU</td>
  </tr>
  <tr>
    <td><code>tensorrt</code></td>
    <td><a href="https://developer.nvidia.com/tensorrt">TensorRT</a>，NVIDIA 提供的高性能深度学习推理库，针对 NVIDIA GPU 进行优化以提升速度。</td>
    <td>GPU</td>
  </tr>
  <tr>
    <td><code>om</code></td>
    <td></td>
    <td>NPU</td>
  </tr>
</table>

`backend_config` 根据不同后端有不同的可选值，如下表所示：

<table>
  <tr>
    <th>后端</th>
    <th>可选值</th>
  </tr>
  <tr>
    <td><code>openvino</code></td>
    <td><code>cpu_num_threads</code>：CPU推理使用的逻辑处理器数量。默认为<code>8</code>。</td>
  </tr>
  <tr>
    <td><code>onnxruntime</code></td>
    <td><code>cpu_num_threads</code>：CPU推理时算子内部的并行计算线程数。默认为<code>8</code>。</td>
  </tr>
  <tr>
    <td><code>tensorrt</code></td>
    <td>
      <code>precision</code>：使用的精度，<code>fp16</code>或<code>fp32</code>。默认为<code>fp32</code>。
      <code>dynamic_shapes</code>：动态形状。
    </td>
  </tr>
  <tr>
    <td><code>om</code></td>
    <td>暂无</td>
  </tr>
</table>

PaddleX 结合模型信息与运行环境信息为每个模型提供默认的高性能推理配置，其中包括推理后端和推理后端的配置。这些默认配置经过精心准备，以便在数个常见场景中可用，且能够取得较优的性能。因此，通常用户可能并不需要关心这些配置的具体细节。

然而，由于实际部署环境与需求的多样性，对于默认配置无法满足要求的情形，用户可以手动调整配置。例如两种常见的情形：

- 更换推理后端。

  模型产线更换推理后端，以通用OCR产线为例：

  <details><summary>👉 <b>1. 修改产线配置文件方式（点击展开）</b></summary>

  ```yaml
  # 支持在不同层级增加配置以实现不同粒度的控制
  # 对于`hpi_config`，子模块或子产线可以覆盖上级配置的顶层字段
  # 使用此方式可以实现“仅产线中的某个子产线/子模块使用高性能推理”

  pipeline_name: OCR

  text_type: general

  use_doc_preprocessor: True
  use_textline_orientation: True

  SubPipelines:
    DocPreprocessor:
      pipeline_name: doc_preprocessor
      use_doc_orientation_classify: True
      use_doc_unwarping: True
      # 当前子产线中的子模块默认启用高性能推理
      use_hpip: True
      # 当前子产线中的子模块默认使用如下高性能推理配置
      hpi_config:
          auto_config: False
          backend: onnxruntime
      SubModules:
        DocOrientationClassify:
          module_name: doc_text_orientation
          model_name: PP-LCNet_x1_0_doc_ori
          model_dir: null
          # 当前子模块不启用高性能推理
          use_hpip: False
        DocUnwarping:
          module_name: image_unwarping
          model_name: UVDoc
          model_dir: null
          # 当前子模块使用如下高性能推理配置
          hpi_config:
              backend: tensorrt

  SubModules:
    TextDetection:
      module_name: text_detection
      model_name: PP-OCRv4_mobile_det
      model_dir: null
      limit_side_len: 960
      limit_type: max
      thresh: 0.3
      box_thresh: 0.6
      unclip_ratio: 2.0
      # 当前子模块启用高性能推理
      use_hpip: True
      # 当前子模块使用如下高性能推理配置
      hpi_config:
          auto_config: False
          backend: onnxruntime
    TextLineOrientation:
      module_name: textline_orientation
      model_name: PP-LCNet_x0_25_textline_ori
      model_dir: null
      batch_size: 6
    TextRecognition:
      module_name: text_recognition
      model_name: PP-OCRv4_mobile_rec
      model_dir: null
      batch_size: 6
      score_thresh: 0.0
  ```

  </details>
  <br />
  <details><summary>👉 <b>2. CLI传参方式（点击展开）</b></summary>

  ```bash
  paddlex \
      --pipeline image_classification \
      --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
      --device gpu:0 \
      --use_hpip \
      --hpi_config '{"auto_config": False, "backend": "onnxruntime"}'
  ```

  </details>
  <br />
  <details><summary>👉 <b>3. Python API传参方式（点击展开）</b></summary>

  ```python
  from paddlex import create_pipeline

  pipeline = create_pipeline(
      pipeline="OCR",
      device="gpu",
      use_hpip=True,
      hpi_config={"auto_config": False, "backend": "onnxruntime"}
  )
  ```

  </details>
  <br />

  单功能模块更换推理后端，以图像分类模块为例：

  <details><summary>👉 <b>1. 修改产线配置文件方式（点击展开）</b></summary>

  ```yaml
  # paddlex/configs/modules/image_classification/ResNet18.yaml
  ...
  Predict:
    ...
    use_hpip: True
    hpi_config:
        auto_config: False
        backend: onnxruntime
    ...
  ...
  ```

  </details>
  <br />
  <details><summary>👉 <b>2. CLI传参方式（点击展开）</b></summary>

  ```bash
  python main.py \
      -c paddlex/configs/modules/image_classification/ResNet18.yaml \
      -o Global.mode=predict \
      -o Predict.model_dir=None \
      -o Predict.input=https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
      -o Global.device=gpu:0 \
      -o Predict.use_hpip=True \
      -o Predict.hpi_config='{"auto_config": False, "backend": "onnxruntime"}'
  ```

  </details>
  <br />
  <details><summary>👉 <b>3. Python API传参方式（点击展开）</b></summary>

  ```python
  from paddlex import create_model

  model = create_model(
      model_name="ResNet18",
      device="gpu",
      use_hpip=True,
      hpi_config={"auto_config": False, "backend": "onnxruntime"}
  )
  ```

  </details>
  <br />

- 修改 Paddle Inference 或 TensorRT 的动态形状配置：

  动态形状是 TensorRT 延迟指定部分或全部张量维度直到运行时的能力。当默认的动态形状配置无法满足需求（例如，模型可能需要范围外的输入形状），用户需要修改相应的配置。

  下面以修改产线配置文件方式为例，CLI传参和Python API传参方式参考更换推理后端中的例子。

  模型产线以通用图像分类产线为例：

  <details><summary>👉 <b>点击展开</b></summary>

  ```yaml
    ...
    SubModules:
      ImageClassification:
        ...
        hpi_config:
          auto_config: False
          backend: tensorrt
          backend_config:
            precision: fp32
            dynamic_shapes:
              x:
                - [1, 3, 300, 300]
                - [4, 3, 300, 300]
                - [32, 3, 1200, 1200]
              ...
    ...
  ```

  </details>
  <br />

  单功能模块以图像分类模块为例：
  <details><summary>👉 <b>点击展开</b></summary>

  ```yaml
  # paddlex/configs/modules/image_classification/ResNet18.yaml
  ...
  Predict:
    ...
    use_hpip: True
    hpi_config:
        auto_config: False
        backend: onnxruntime
        backend_config:
          precision: fp32
          dynamic_shapes:
            x:
              - [1, 3, 300, 300]
              - [4, 3, 300, 300]
              - [32, 3, 1200, 1200]
    ...
  ...
  ```

  </details>
  <br />

  在 `dynamic_shapes` 中，需要为每一个输入张量指定动态形状，格式为：`{输入张量名称}: [{最小形状}, [{最优形状}], [{最大形状}]]`。有关最小形状、最优形状以及最大形状的相关介绍及更多细节，请参考 [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)。

  在完成修改后，请删除模型目录中的缓存文件（`shape_range_info.pbtxt` 与 `trt_serialized` 开头的文件）。

  关于修改 Paddle-TensorRT 的动态形状的具体方法，请参考 [PaddleX单模型Python脚本使用说明: 4. 推理后端设置](../module_usage/instructions/model_python_API.md)。

### 2.2 自定义编译高性能推理插件

高性能推理插件 `ultra-infer` 位于 `PaddleX/libs/ultra-infer` 目录。编译脚本位于 `PaddleX/libs/ultra-infer/scripts/linux/set_up_docker_and_build_py.sh` ，编译默认编译GPU版本和包含 `OpenVINO`、`TensorRT`、`ONNX Runtime` 三种推理后端的 `ultra-infer`。

编译示例：

```shell
# 编译
# export PYTHON_VERSION=...
# export WITH_GPU=...
# export ENABLE_ORT_BACKEND=...
# export ...

cd PaddleX/libs/ultra-infer/scripts/linux
bash set_up_docker_and_build_py.sh

# 安装
python -m pip install ../../python/dist/ultra_infer*.whl
```

编译时可根据需求修改如下选项：

<table>
    <thead>
        <tr>
            <th>选项</th>
            <th>说明</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>http_proxy</td>
            <td>在下载三方库时使用具体的http代理，默认空</td>
        </tr>
        <tr>
            <td>PYTHON_VERSION</td>
            <td>Python版本，默认 <code>3.10.0</code></td>
        </tr>
        <tr>
            <td>WITH_GPU</td>
            <td>是否编译支持Nvidia-GPU，默认 <code>ON</code></td>
        </tr>
        <tr>
            <td>ENABLE_ORT_BACKEND</td>
            <td>是否编译集成ONNX Runtime后端，默认 <code>ON</code></td>
        </tr>
        <tr>
            <td>ENABLE_TRT_BACKEND</td>
            <td>是否编译集成TensorRT后端（仅支持GPU），默认 <code>ON</code></td>
        </tr>
        <tr>
            <td>ENABLE_OPENVINO_BACKEND</td>
            <td>是否编译集成OpenVINO后端（仅支持CPU），默认 <code>ON</code></td>
        </tr>
    </tbody>
</table>

## 3. 常见问题

1. 为什么使用高性能推理功能后，推理速度还是与普通推理的速度差不多？

- 高性能推理通过智能选择后端来加速推理，但由于模型复杂性或不支持算子等情况，部分模型可能无法使用加速后端（如OpenVINO、TensorRT等）。此时会选择已知**最快的可用后端**，因此可能退回到普通推理。

2. 高性能推理功能是否支持所有模型产线与单功能模块？

- 高性能推理功能支持所有模型产线与单功能模块，但部分模型可能无法加速推理，此时日志中会提示相关内容，具体原因可以参考问题1。

3. 为什么安装高性能推理插件会失败？

- 高性能推理功能目前支持的环境如 [1.1节的表](#11-安装高性能推理插件) 所示。如果安装失败，可能是高性能推理功能不支持当前环境。另外，CUDA 12.6 已经在支持中。

4. 为什么使用高性能推理功能后，程序在运行过程中会卡住或者弹出一些 WARNING 和 ERROR 信息？这种情况下应该如何处理？

- 在引擎构建过程中，由于子图优化和算子处理，可能会导致程序耗时较长，并生成一些 WARNING 和 ERROR 信息。然而，只要程序没有自动退出，建议耐心等待，程序通常会继续运行至完成。
