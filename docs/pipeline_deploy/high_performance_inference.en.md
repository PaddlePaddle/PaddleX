---
comments: true
---

# PaddleX High-Performance Inference Guide

In actual production environments, many applications have stringent standards for the performance metrics of deployment strategies (especially response speed) to ensure efficient system operation and smooth user experiences. To this end, PaddleX provides a high-performance inference plugin that significantly improves model inference speed for users without requiring them to focus on complex configurations and low-level details, through automatic configuration and multi-backend inference capabilities.

## Table of Contents

- [1. Basic Usage](#1.-basic-usage)
  - [1.1 Installing the High-Performance Inference Plugin](#11-installing-the-high-performance-inference-plugin)
  - [1.2 Enabling High-Performance Inference](#12-enabling-high-performance-inference)
- [2. Advanced Usage](#2-advanced-usage)
  - [2.1 High-Performance Inference Modes](#21-high-performance-inference-modes)
  - [2.2 High-Performance Inference Configuration](#22-high-performance-inference-configuration)
  - [2.3 Modifying High-Performance Inference Configuration](#23-modifying-high-performance-inference-configuration)
  - [2.4 Example of Modifying High-Performance Inference Configuration](#24-example-of-modifying-high-performance-inference-configuration)
  - [2.5 Enabling/Disabling High-Performance Inference in Sub-pipelines/Sub-modules](#25-enablingdisabling-high-performance-inference-in-sub-pipelinessub-modules)
  - [2.6 Model Caching Instructions](#26-model-caching-instructions)
  - [2.7 Customizing Model Inference Libraries](#27-customizing-model-inference-libraries)
- [3. Frequently Asked Questions](#3.-frequently-asked-questions)

## 1. Basic Usage

Before using the high-performance inference plugin, ensure you have completed the installation of PaddleX according to the [PaddleX Local Installation Tutorial](../installation/installation.en.md) and successfully run the quick inference using the PaddleX pipeline command-line instructions or Python script instructions.

High-performance inference supports processing PaddlePaddle static models( `.pdmodel`, `.json` ) and ONNX format models( `.onnx` )**. For ONNX format models, it is recommended to use the [Paddle2ONNX plugin](./paddle2onnx.en.md) for conversion. If multiple format models exist in the model directory, PaddleX will automatically select them as needed.

### 1.1 Installing the High-Performance Inference Plugin

The processor architectures, operating systems, device types, and Python versions currently supported by high-performance inference are shown in the table below:

<table>
  <tr>
    <th>Operating System</th>
    <th>Processor Architecture</th>
    <th>Device Type</th>
    <th>Python Version</th>
  </tr>
  <tr>
    <td rowspan="5">Linux</td>
    <td rowspan="4">x86-64</td>
  </tr>
  <tr>
    <td>CPU</td>
    <td>3.8–3.12</td>
  </tr>
  <tr>
    <td>GPU (CUDA 11.8 + cuDNN 8.9)</td>
    <td>3.8–3.12</td>
  </tr>
  <tr>
    <td>NPU</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>aarch64</td>
    <td>NPU</td>
    <td>3.10</td>
  </tr>
</table>

#### (1) Installing the High-Performance Inference Plugin Based on Docker (Highly Recommended):

Refer to [Get PaddleX based on Docker](../installation/installation.en.md#21-obtaining-paddlex-based-on-docker) to use Docker to start the PaddleX container. After starting the container, execute the following commands according to the device type to install the high-performance inference plugin:

<table>
    <thead>
        <tr>
            <th>Device Type</th>
            <th>Installation Command</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CPU</td>
            <td><code>paddlex --install hpi-cpu</code></td>
            <td>Installs the CPU version of high-performance inference.</td>
        </tr>
        <tr>
            <td>GPU</td>
            <td><code>paddlex --install hpi-gpu</code></td>
            <td>Installs the GPU version of high-performance inference.<br />Includes all features of the CPU version.</td>
        </tr>
        <tr>
            <td>NPU</td>
            <td><code>paddlex --install hpi-npu</code></td>
            <td>Installs the NPU version of high-performance inference.<br />For usage instructions, please refer to the <a href="../practical_tutorials/high_performance_npu_tutorial.en.md">Ascend NPU High-Performance Inference Tutorial</a>.</td>
        </tr>
    </tbody>
</table>

#### (2) Local Installation of High-Performance Inference Plugin:

##### Installing the High-Performance Inference Plugin for CPU:

Execute:

```bash
paddlex --install hpi-cpu
```

##### Installing the High-Performance Inference Plugin for GPU:

Refer to the [NVIDIA official website](https://developer.nvidia.com/) to install CUDA and cuDNN locally, then execute:

```bash
paddlex --install hpi-gpu
```

The required CUDA and cuDNN versions can be obtained through the following commands:

```bash
# CUDA version
pip list | grep nvidia-cuda
# cuDNN version
pip list | grep nvidia-cudnn
```

Reference documents for installing CUDA 11.8 and cuDNN 8.9:
- [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [Install cuDNN 8.9](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html)

**Notes**:

1. **GPUs only support CUDA 11.8 + cuDNN 8.9**, and support for CUDA 12.6 is under development.

2. Only one version of the high-performance inference plugin should exist in the same environment.

3. For instructions on high-performance inference using NPU devices, refer to the [Ascend NPU High-Performance Inference Tutorial](../practical_tutorials/high_performance_npu_tutorial.md).

4. Windows only supports installing and using the high-performance inference plugin via Docker.

### 1.2 Enabling High-Performance Inference

Below are examples of enabling high-performance inference in the general image classification pipeline and image classification module using PaddleX CLI and Python API.

For PaddleX CLI, specify `--use_hpip` to enable high-performance inference.

General Image Classification Pipeline:

```bash
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip
```

Image Classification Module:

```bash
python main.py \
    -c paddlex/configs/modules/image_classification/ResNet18.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.input=https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    -o Global.device=gpu:0 \
    -o Predict.use_hpip=True
```

For the PaddleX Python API, the method to enable high-performance inference is similar. Taking the General Image Classification Pipeline and Image Classification Module as examples:

General Image Classification Pipeline:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="image_classification",
    device="gpu",
    use_hpip=True
)

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

Image Classification Module:

```python
from paddlex import create_model

model = create_model(
    model_name="ResNet18",
    device="gpu",
    use_hpip=True
)

output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

The inference results obtained with the high-performance inference plugin enabled are consistent with those without the plugin. For some models, **it may take a longer time to complete the construction of the inference engine when enabling the high-performance inference plugin for the first time**. PaddleX will cache relevant information in the model directory after the first construction of the inference engine and reuse the cached content in subsequent runs to improve initialization speed.

**Enabling high-performance inference by default affects the entire pipeline/module**. If you want to control the scope of application with finer granularity, such as enabling the high-performance inference plugin for only a specific sub-pipeline or sub-module within the pipeline, you can set `use_hpip` at different levels of configuration in the pipeline configuration file. Please refer to [2.5 Enabling/Disabling High-Performance Inference in Sub-pipelines/Sub-modules](#25-enablingdisabling-high-performance-inference-in-sub-pipelinessub-modules).

## 2. Advanced Usage

This section introduces the advanced usage of high-performance inference, suitable for users who have some understanding of model deployment or wish to manually configure and optimize. Users can customize the use of high-performance inference based on their own needs by referring to the configuration instructions and examples. Next, the advanced usage methods will be introduced in detail.

### 2.1 High-Performance Inference Modes

High-performance inference is divided into two modes:

#### (1) Safe Auto-Configuration Mode

The safe auto-configuration mode has a protection mechanism and **automatically selects the configuration with better performance for the current environment by default**. In this mode, users can override the default configuration, but the provided configuration will be checked, and PaddleX will reject unavailable configurations based on prior knowledge. This is the default mode.

#### (2) Unrestricted Manual Configuration Mode

The unrestricted manual configuration mode provides complete configuration freedom, allowing **free selection of the inference backend and modification of backend configurations**, but cannot guarantee successful inference. This mode is suitable for experienced users with specific needs for the inference backend and its configurations and is recommended for use after familiarizing with high-performance inference.

### 2.2 High-Performance Inference Configuration

Common high-performance inference configurations include the following fields:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>auto_config</code></td>
<td>Whether to enable the safe auto-configuration mode.<br /><code>True</code> to enable, <code>False</code> to enable the unrestricted manual configuration mode.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
  <td><code>backend</code></td>
  <td>Specifies the inference backend to use. Cannot be <code>None</code> in unrestricted manual configuration mode.</td>
  <td><code>str | None</code></td>
  <td><code>None</code></td>
</tr>
<tr>
  <td><code>backend_config</code></td>
  <td>The configuration of the inference backend, which can override the default configuration items of the backend if it is not <code>None</code>.</td>
  <td><code>dict | None</code></td>
  <td><code>None</code></td>
</tr>
<tr>
  <td><code>auto_paddle2onnx</code></td>
  <td>Whether to enable the <a href="./paddle2onnx.en.md">Paddle2ONNX plugin</a> to automatically convert Paddle models to ONNX models.</td>
  <td><code>bool</code></td>
  <td><code>True</code></td>
</tr>
</tbody>
</table>

The available options for `backend` are shown in the following table:

<table>
  <tr>
    <th>Option</th>
    <th>Description</th>
    <th>Supported Devices</th>
  </tr>
  <tr>
    <td><code>paddle</code></td>
    <td>Paddle Inference engine, supporting the Paddle Inference TensorRT subgraph engine to improve GPU inference performance of models.</td>
    <td>CPU, GPU</td>
  </tr>
  <tr>
    <td><code>openvino</code></td>
    <td><a href="https://github.com/openvinotoolkit/openvino">OpenVINO</a>, a deep learning inference tool provided by Intel, optimized for model inference performance on various Intel hardware.</td>
    <td>CPU</td>
  </tr>
  <tr>
    <td><code>onnxruntime</code></td>
    <td><a href="https://onnxruntime.ai/">ONNX Runtime</a>, a cross-platform, high-performance inference engine.</td>
    <td>CPU, GPU</td>
  </tr>
  <tr>
    <td><code>tensorrt</code></td>
    <td><a href="https://developer.nvidia.com/tensorrt">TensorRT</a>, a high-performance deep learning inference library provided by NVIDIA, optimized for NVIDIA GPUs to improve speed.</td>
    <td>GPU</td>
  </tr>
  <tr>
    <td><code>om</code></td>
    <td>a inference engine of offline model format customized for Huawei Ascend NPU, deeply optimized for hardware to reduce operator computation time and scheduling time, effectively improving inference performance.</td>
    <td>NPU</td>
  </tr>
</table>

The available values for `backend_config` vary depending on the backend, as shown in the following table:

<table>
  <tr>
    <th>Backend</th>
    <th>Available Values</th>
  </tr>
  <tr>
    <td><code>paddle</code></td>
    <td>Refer to <a href="../module_usage/instructions/model_python_API.en.md">PaddleX Single Model Python Usage Instructions: 4. Inference Backend Configuration</a>.</td>
  </tr>
  <tr>
    <td><code>openvino</code></td>
    <td><code>cpu_num_threads</code>: The number of logical processors used for CPU inference. Default is <code>8</code>.</td>
  </tr>
  <tr>
    <td><code>onnxruntime</code></td>
    <td><code>cpu_num_threads</code>: The number of parallel computing threads within operators for CPU inference. Default is <code>8</code>.</td>
  </tr>
  <tr>
    <td><code>tensorrt</code></td>
    <td>
      <code>precision</code>: The precision used, <code>fp16</code> or <code>fp32</code>. Default is <code>fp32</code>.
      <br />
      <code>dynamic_shapes</code>: Dynamic shapes. Dynamic shapes include minimum shape, optimal shape, and maximum shape, which represent TensorRT’s ability to defer specifying some or all tensor dimensions until runtime. The format is:<code>{input tensor name}: [{minimum shape}, [{optimal shape}], [{maximum shape}]]</code>. For more information, please refer to the  <a href="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes">TensorRT official documentation.</a>。
  <tr>
    <td><code>om</code></td>
    <td>None</td>
  </tr>
</table>

### 2.3 How to Modify High-Performance Inference Configuration

Due to the diversity of actual deployment environments and requirements, the default configuration may not meet all needs. In such cases, manual adjustments to the high-performance inference configuration may be necessary. Here are two common scenarios:

- Needing to change the inference backend.
  - For example, in an OCR pipeline, specifying the `text_detection` module to use the `onnxruntime` backend and the `text_recognition` module to use the `tensorrt` backend.

- Needing to modify the dynamic shape configuration for TensorRT:
  - When the default dynamic shape configuration cannot meet requirements (e.g., the model may require input shapes outside the specified range), dynamic shapes need to be specified for each input tensor. After modification, the model's `.cache` directory should be cleaned up.

In these scenarios, users can modify the configuration by altering the `hpi_config` field in the **pipeline/module configuration file**, **CLI** parameters, or **Python API** parameters. **Parameters passed through CLI or Python API will override settings in the pipeline/module configuration file**.

### 2.4 Examples of Modifying High-Performance Inference Configuration

#### (1) Changing the Inference Backend

##### Using the `onnxruntime` backend for all models in a general OCR pipeline:

<details><summary>👉 1. Modifying the pipeline configuration file (click to expand)</summary>

```yaml
pipeline_name: OCR

use_hpip: True
hpi_config:
    backend: onnxruntime

...
```

</details>
<details><summary>👉 2. CLI parameter passing method (click to expand)</summary>

```bash
paddlex \
      --pipeline image_classification \
      --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
      --device gpu:0 \
      --use_hpip \
      --hpi_config '{"backend": "onnxruntime"}'
```

</details>
<details><summary>👉 3. Python API parameter passing method (click to expand)</summary>

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
      pipeline="OCR",
      device="gpu",
      use_hpip=True,
      hpi_config={"backend": "onnxruntime"}
)
```

</details>

##### Using the `onnxruntime` backend for the image classification module:

<details><summary>👉 1. Modifying the module configuration file (click to expand)</summary>

```yaml
# paddlex/configs/modules/image_classification/ResNet18.yaml
...
Predict:
    ...
    use_hpip: True
    hpi_config:
        backend: onnxruntime
    ...
...
```

</details>
<details><summary>👉 2. CLI parameter passing method (click to expand)</summary>

```bash
python main.py \
      -c paddlex/configs/modules/image_classification/ResNet18.yaml \
      -o Global.mode=predict \
      -o Predict.model_dir=None \
      -o Predict.input=https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
      -o Global.device=gpu:0 \
      -o Predict.use_hpip=True \
      -o Predict.hpi_config='{"backend": "onnxruntime"}'
```

</details>
<details><summary>👉 3. Python API parameter passing method (click to expand)</summary>

```python
from paddlex import create_model

model = create_model(
      model_name="ResNet18",
      device="gpu",
      use_hpip=True,
      hpi_config={"backend": "onnxruntime"}
)
```

</details>

##### Using the `onnxruntime` backend for the `text_detection` module and the `tensorrt` backend for the `text_recognition` module in a general OCR pipeline:

<details><summary>👉 1. Modifying the pipeline configuration file (click to expand)</summary>

```yaml
pipeline_name: OCR

...

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
      # Enable high-performance inference for the current submodule
      use_hpip: True
      # High-performance inference configuration for the current submodule
      hpi_config:
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
      # Enable high-performance inference for the current submodule
      use_hpip: True
      # High-performance inference configuration for the current submodule
      hpi_config:
          backend: tensorrt
```

</details>

#### (2) Modify TensorRT's Dynamic Shape Configuration

##### Modifying dynamic shape configuration for general image classification pipeline:

<details><summary>👉 Click to Expand</summary>

```yaml
    ...
    SubModules:
      ImageClassification:
        ...
        hpi_config:
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

##### Modifying dynamic shape configuration for image classification module:

<details><summary>👉 Click to Expand</summary>

```yaml
...
Predict:
    ...
    use_hpip: True
    hpi_config:
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

### 2.5 Enabling/Disabling High-Performance Inference in Sub-pipelines/Sub-modules

High-performance inference support allows **only specific sub-pipelines/sub-modules within a pipeline to use high-performance inference** by utilizing `use_hpip` at the sub-pipeline/sub-module level. Examples are as follows:

##### Enabling High-Performance Inference for the `text_detection` module in general OCR pipeline, while disabling it for the `text_recognition` module:

<details><summary>👉 Click to Expand</summary>

```yaml
pipeline_name: OCR

...

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
      use_hpip: True # Enable high-performance inference for the current sub-module
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
      use_hpip: False # Disable high-performance inference for the current sub-module
```

</details>

**Notes**:

1. When setting `use_hpip` in a sub-pipeline or sub-module, the deepest-level configuration takes precedence.

2. **It is strongly recommended to enable high-performance inference by modifying the pipeline configuration file**, rather than using CLI or Python API settings. Enabling `use_hpip` through CLI or Python API is equivalent to setting `use_hpip` at the top level of the configuration file.

### 2.6 Model Cache Description

The model cache will be stored in the `.cache` directory under the model directory, including files such as `shape_range_info.pbtxt` and those prefixed with `trt_serialized` generated when using the `tensorrt` or `paddle` backend.

When the `auto_paddle2onnx` option is enabled, an `inference.onnx` file may be automatically generated in the model directory.

### 2.7 Custom Model Inference Library

`ultra-infer` is the underlying model inference library for high-performance inference, located in the `PaddleX/libs/ultra-infer` directory. The compilation script is located at `PaddleX/libs/ultra-infer/scripts/linux/set_up_docker_and_build_py.sh`. The default compilation builds the GPU version and includes OpenVINO, TensorRT, and ONNX Runtime as inference backends for `ultra-infer`.

When compiling customized versions, you can modify the following options as needed:

<table>
    <thead>
        <tr>
            <th>Option</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>http_proxy</td>
            <td>Use a specific HTTP proxy when downloading third-party libraries, default is empty</td>
        </tr>
        <tr>
            <td>PYTHON_VERSION</td>
            <td>Python version, default is <code>3.10.0</code></td>
        </tr>
        <tr>
            <td>WITH_GPU</td>
            <td>Whether to compile support for Nvidia-GPU, default is <code>ON</code></td>
        </tr>
        <tr>
            <td>ENABLE_ORT_BACKEND</td>
            <td>Whether to compile and integrate the ONNX Runtime backend, default is <code>ON</code></td>
        </tr>
        <tr>
            <td>ENABLE_TRT_BACKEND</td>
            <td>Whether to compile and integrate the TensorRT backend (GPU only), default is <code>ON</code></td>
        </tr>
        <tr>
            <td>ENABLE_OPENVINO_BACKEND</td>
            <td>Whether to compile and integrate the OpenVINO backend (CPU only), default is <code>ON</code></td>
        </tr>
    </tbody>
</table>

Compilation Example:

```shell
# Compilation
# export PYTHON_VERSION=...
# export WITH_GPU=...
# export ENABLE_ORT_BACKEND=...
# export ...

cd PaddleX/libs/ultra-infer/scripts/linux
bash set_up_docker_and_build_py.sh

# Installation
python -m pip install ../../python/dist/ultra_infer*.whl
```

## 3. Frequently Asked Questions

**1. Why is the inference speed similar to regular inference after using the high-performance inference feature?**

High-performance inference accelerates inference by intelligently selecting backends, but due to factors such as model complexity or unsupported operators, some models may not be able to use accelerated backends (like OpenVINO, TensorRT, etc.). In such cases, relevant information will be prompted in the logs, and the **fastest available backend** known will be selected, potentially reverting to regular inference.

The high-performance inference plugin accelerates inference by intelligently selecting the backend.

For modules, due to model complexity or unsupported operators, some models may not be able to use accelerated backends (such as OpenVINO, TensorRT, etc.). In such cases, relevant information will be prompted in the logs, and the **fastest available backend** known will be selected, potentially falling back to regular inference.

For pipelines, the performance bottleneck may not be in the model inference stage.

You can use the [PaddleX benchmark](../module_usage/instructions/benchmark.md) tool to conduct actual speed tests for a more accurate performance assessment.

**2. Does the high-performance inference feature support all model pipelines and modules?**

The high-performance inference feature supports all model pipelines and modules, but some models may not experience accelerated inference. Specific reasons can be referred to in Question 1.

**3. Why does the installation of the high-performance inference plugin fail, with the log displaying: "Currently, the CUDA version must be 11.x for GPU devices."?**

The environments supported by the high-performance inference feature are shown in [the table in Section 1.1](#11-installing-the-high-performance-inference-plugin). If the installation fails, it may be due to the high-performance inference feature not supporting the current environment. Additionally, CUDA 12.6 is already under support.

**4. Why does the program get stuck or display WARNING and ERROR messages when using the high-performance inference feature? How should this be handled?**

During engine construction, due to subgraph optimization and operator processing, the program may take longer and generate WARNING and ERROR messages. However, as long as the program does not exit automatically, it is recommended to wait patiently as the program will usually continue to run until completion.
