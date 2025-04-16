---
comments: true
---

# PaddleX High-Performance Inference Guide

In real production environments, many applications impose strict performance metrics—especially in response time—on deployment strategies to ensure system efficiency and a smooth user experience. To address this, PaddleX offers a high-performance inference plugin that, through automatic configuration and multi-backend inference capabilities, enables users to significantly accelerate model inference without concerning themselves with complex configurations and low-level details.

## Table of Contents

- [1. Basic Usage](#1.-Basic-Usage)
  - [1.1 Installing the High-Performance Inference Plugin](#1.1-Installing-the-High-Performance-Inference-Plugin)
  - [1.2 Enabling the High-Performance Inference Plugin](#1.2-Enabling-the-High-Performance-Inference-Plugin)
- [2. Advanced Usage](#2-Advanced-Usage)
  - [2.1 Working Modes of High-Performance Inference](#21-Working-Modes-of-High-Performance-Inference)
  - [2.2 High-Performance Inference Configuration](#22-High-Performance-Inference-Configuration)
  - [2.3 Modifying the High-Performance Inference Configuration](#23-Modifying-the-High-Performance-Inference-Configuration)
  - [2.4 Enabling/Disabling the High-Performance Inference Plugin on Sub-pipelines/Submodules](#24-EnablingDisabling-the-High-Performance-Inference-Plugin-on-Sub-pipelinesSubmodules)
  - [2.5 Model Cache Description](#25-Model-Cache-Description)
  - [2.6 Customizing the Model Inference Library](#26-Customizing-the-Model-Inference-Library)
- [3. Frequently Asked Questions](#3-Frequently-Asked-Questions)

## 1. Basic Usage

Before using the high-performance inference plugin, please ensure that you have completed the PaddleX installation according to the [PaddleX Local Installation Tutorial](../installation/installation.en.md) and have run the quick inference using the PaddleX pipeline command line or the PaddleX pipeline Python script as described in the usage instructions.

High-performance inference supports handling **PaddlePaddle static graph models (`.pdmodel`, `.json`)** and **ONNX format models (`.onnx`)**. For ONNX format models, it is recommended to convert them using the [Paddle2ONNX Plugin](./paddle2onnx.md). If multiple model formats are present in the model directory, PaddleX will automatically choose the appropriate one as needed.

### 1.1 Installing the High-Performance Inference Plugin

Currently, the supported processor architectures, operating systems, device types, and Python versions for high-performance inference are as follows:

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
    <td>GPU&nbsp;(CUDA&nbsp;11.8&nbsp;+&nbsp;cuDNN&nbsp;8.9)</td>
    <td>3.8–3.12</td>
  </tr>
  <tr>
    <td>NPU</td>
    <td>3.10</td>
  </tr>
  <tr>
    <td>AArch64</td>
    <td>NPU</td>
    <td>3.10</td>
  </tr>
</table>

#### (1) Installing the High-Performance Inference Plugin in a Docker Container (Highly Recommended):

Refer to [Get PaddleX based on Docker](../installation/installation.en.md#21-obtaining-paddlex-based-on-docker) to start a PaddleX container using Docker. After starting the container, execute the following commands according to your device type to install the high-performance inference plugin:

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
          <td>Installs the CPU version of the high-performance inference functionality.</td>
      </tr>
      <tr>
          <td>GPU</td>
          <td><code>paddlex --install hpi-gpu</code></td>
          <td>Installs the GPU version of the high-performance inference functionality.<br />Includes all functionalities of the CPU version.</td>
      </tr>
  </tbody>
</table>

In the official PaddleX Docker image, TensorRT is installed by default. The high-performance inference plugin can then accelerate inference using the Paddle Inference TensorRT subgraph engine.

**Please note that the aforementioned Docker image refers to the official PaddleX image described in [Get PaddleX via Docker](../installation/installation.en.md#21-get-paddlex-based-on-docker), rather than the PaddlePaddle official image described in [PaddlePaddle Local Installation Tutorial](../installation/paddlepaddle_install.en.md#installing-paddlepaddle-via-docker). For the latter, please refer to the local installation instructions for the high-performance inference plugin.**

#### (2) Installing the High-Performance Inference Plugin Locally (Not Recommended):

##### To install the CPU version of the high-performance inference plugin:

Run:

```bash
paddlex --install hpi-cpu
```

##### To install the GPU version of the high-performance inference plugin:

Before installation, please ensure that CUDA and cuDNN are installed in your environment. The official PaddleX currently only provides a precompiled package for CUDA 11.8 + cuDNN 8.9, so please ensure that the installed versions of CUDA and cuDNN are compatible with the compiled versions. Below are the installation documentation links for CUDA 11.8 and cuDNN 8.9:

- [Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- [Install cuDNN 8.9](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-890/install-guide/index.html)

If you are using the official PaddlePaddle image, the CUDA and cuDNN versions in the image already meet the requirements, so there is no need for a separate installation.

If PaddlePaddle is installed via pip, the relevant CUDA and cuDNN Python packages will usually be installed automatically. In this case, **you still need to install the non-Python-specific CUDA and cuDNN**. It is also advisable to install the CUDA and cuDNN versions that match the versions of the Python packages in your environment to avoid potential issues arising from coexisting libraries of different versions. You can check the versions of the CUDA and cuDNN related Python packages as follows:

```bash
# For CUDA related Python packages
pip list | grep nvidia-cuda
# For cuDNN related Python packages
pip list | grep nvidia-cudnn
```

If you wish to use the Paddle Inference TensorRT subgraph engine, you will need to install TensorRT additionally. Please refer to the related instructions in the [PaddlePaddle Local Installation Tutorial](../installation/paddlepaddle_install.en.md). Note that because the underlying inference library of the high-performance inference plugin also integrates TensorRT, it is recommended to install the same version of TensorRT to avoid version conflicts. Currently, the TensorRT version integrated into the high-performance inference plugin's underlying inference library is 8.6.1.6. If you are using the official PaddlePaddle image, you do not need to worry about version conflicts.

After confirming that the correct versions of CUDA, cuDNN, and TensorRT (optional) are installed, run:

```bash
paddlex --install hpi-gpu
```

##### To install the NPU version of the high-performance inference plugin:

Please refer to the [Ascend NPU High-Performance Inference Tutorial](../practical_tutorials/high_performance_npu_tutorial.md).

**Note:**

1. **Currently, the official PaddleX only provides a precompiled package for CUDA 11.8 + cuDNN 8.9**; support for CUDA 12.6 is in progress.
2. Only one version of the high-performance inference plugin should exist in the same environment.
3. For Windows systems, it is currently recommended to install and use the high-performance inference plugin within a Docker container.

### 1.2 Enabling the High-Performance Inference Plugin

Below are examples of enabling the high-performance inference plugin in both the PaddleX CLI and Python API for the general image classification pipeline and the image classification module.

For the PaddleX CLI, specify `--use_hpip` to enable the high-performance inference plugin.

**General Image Classification Pipeline:**

```bash
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip
```

**Image Classification Module:**

```bash
python main.py \
    -c paddlex/configs/modules/image_classification/ResNet18.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir=None \
    -o Predict.input=https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    -o Global.device=gpu:0 \
    -o Predict.use_hpip=True
```

For the PaddleX Python API, enabling the high-performance inference plugin is similar. For example:

**General Image Classification Pipeline:**

```python
from paddlex import create_pipeline

pipeline = create_pipeline(
    pipeline="image_classification",
    device="gpu",
    use_hpip=True
)

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

**Image Classification Module:**

```python
from paddlex import create_model

model = create_model(
    model_name="ResNet18",
    device="gpu",
    use_hpip=True
)

output = model.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
```

The inference results obtained with the high-performance inference plugin enabled are identical to those without the plugin. For some models, **the first time the high-performance inference plugin is enabled, it may take a longer time to complete the construction of the inference engine**. PaddleX caches the related information in the model directory after the inference engine is built for the first time, and subsequently reuses the cached content to improve the initialization speed.

**By default, enabling the high-performance inference plugin applies to the entire pipeline/module.** If you want to control the scope in a more granular way (e.g., enabling the high-performance inference plugin for only a sub-pipeline or a submodule), you can set the `use_hpip` parameter at different configuration levels in the pipeline configuration file. Please refer to [2.4 Enabling/Disabling the High-Performance Inference Plugin on Sub-pipelines/Submodules](#24-EnablingDisabling-the-High-Performance-Inference-Plugin-on-Sub-pipelinesSubmodules) for more details.

## 2. Advanced Usage

This section introduces the advanced usage of the high-performance inference plugin, which is suitable for users who have a good understanding of model deployment or wish to manually adjust configurations. Users can customize the use of the high-performance inference plugin according to their requirements by referring to the configuration instructions and examples. The following sections describe advanced usage in detail.

### 2.1 Working Modes of High-Performance Inference

The high-performance inference plugin supports two working modes. The operating mode can be switched by modifying the high-performance inference configuration.

#### (1) Safe Auto-Configuration Mode

In safe auto-configuration mode, a protective mechanism is enabled. By default, **the configuration with the best performance for the current environment is automatically selected**. In this mode, while the user can override the default configuration, the provided configuration will be subject to checks, and PaddleX will reject configurations that are not available based on prior knowledge. This is the default operating mode.

#### (2) Unrestricted Manual Configuration Mode

In unrestricted manual configuration mode, full freedom is provided to configure—users can **choose the inference backend freely and modify its configuration, etc.**—but there is no guarantee that inference will always succeed. This mode is recommended for experienced users who have clear requirements for the inference backend and its configuration; it is advised to use this mode only when familiar with high-performance inference.

### 2.2 High-Performance Inference Configuration

Common configuration fields for high-performance inference include:

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
<td>Whether to enable the safe auto-configuration mode.<br /><code>True</code> enables safe auto-configuration mode, <code>False</code> enables the unrestricted manual configuration mode.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
  <td><code>backend</code></td>
  <td>Specifies the inference backend to use. In unrestricted manual configuration mode, it cannot be <code>None</code>.</td>
  <td><code>str | None</code></td>
  <td><code>None</code></td>
</tr>
<tr>
  <td><code>backend_config</code></td>
  <td>The configuration for the inference backend. If not <code>None</code>, it can override the default backend configuration options.</td>
  <td><code>dict | None</code></td>
  <td><code>None</code></td>
</tr>
<tr>
  <td><code>auto_paddle2onnx</code></td>
  <td>Whether to enable the <a href="./paddle2onnx.en.md">Paddle2ONNX plugin</a> to automatically convert a Paddle model to an ONNX model.</td>
  <td><code>bool</code></td>
  <td><code>True</code></td>
</tr>
</tbody>
</table>

The optional values for `backend` are as follows:

<table>
  <tr>
    <th>Option</th>
    <th>Description</th>
    <th>Supported Devices</th>
  </tr>
  <tr>
    <td><code>paddle</code></td>
    <td>Paddle Inference engine; supports enhancing GPU inference performance using the Paddle Inference TensorRT subgraph engine.</td>
    <td>CPU, GPU</td>
  </tr>
  <tr>
    <td><code>openvino</code></td>
    <td><a href="https://github.com/openvinotoolkit/openvino">OpenVINO</a>, a deep learning inference tool provided by Intel, optimized for inference performance on various Intel hardware.</td>
    <td>CPU</td>
  </tr>
  <tr>
    <td><code>onnxruntime</code></td>
    <td><a href="https://onnxruntime.ai/">ONNX Runtime</a>, a cross-platform, high-performance inference engine.</td>
    <td>CPU, GPU</td>
  </tr>
  <tr>
    <td><code>tensorrt</code></td>
    <td><a href="https://developer.nvidia.com/tensorrt">TensorRT</a>, a high-performance deep learning inference library provided by NVIDIA, optimized for NVIDIA GPUs to enhance speed.</td>
    <td>GPU</td>
  </tr>
  <tr>
    <td><code>om</code></td>
    <td>The inference engine corresponding to the offline model format customized for Huawei Ascend NPU, deeply optimized for hardware to reduce operator computation and scheduling time, effectively enhancing inference performance.</td>
    <td>NPU</td>
  </tr>
</table>

The available options for `backend_config` vary for different backends, as shown in the following table:

<table>
  <tr>
    <th>Backend</th>
    <th>Options</th>
  </tr>
  <tr>
    <td><code>paddle</code></td>
    <td>Refer to <a href="../module_usage/instructions/model_python_API.en.md">PaddleX Single-Model Python Script Usage: 4. Inference Backend Settings</a>.</td>
  </tr>
  <tr>
    <td><code>openvino</code></td>
    <td><code>cpu_num_threads</code>: The number of logical processors used for CPU inference. The default is <code>8</code>.</td>
  </tr>
  <tr>
    <td><code>onnxruntime</code></td>
    <td><code>cpu_num_threads</code>: The number of parallel computation threads within the operator during CPU inference. The default is <code>8</code>.</td>
  </tr>
  <tr>
    <td><code>tensorrt</code></td>
    <td>
      <code>precision</code>: The precision used, either <code>fp16</code> or <code>fp32</code>. The default is <code>fp32</code>.
      <br />
      <code>dynamic_shapes</code>: Dynamic shapes. Dynamic shapes include the minimum shape, optimal shape, and maximum shape, which represent TensorRT’s ability to delay specifying some or all tensor dimensions until runtime. The format is: <code>{input tensor name}: [ [minimum shape], [optimal shape], [maximum shape] ]</code>. For more details, please refer to the <a href="https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes">TensorRT official documentation</a>.
    </td>
  </tr>
  <tr>
    <td><code>om</code></td>
    <td>None at the moment</td>
  </tr>
</table>

### 2.3 Modifying the High-Performance Inference Configuration

Due to the diversity of actual deployment environments and requirements, the default configuration might not meet all needs. In such cases, manual adjustment of the high-performance inference configuration may be necessary. Users can modify the configuration by editing the **pipeline/module configuration file** or by passing the `hpi_config` field in the parameters via **CLI** or **Python API**. **Parameters passed via CLI or Python API will override the settings in the pipeline/module configuration file.** The following examples illustrate how to modify the configuration.

#### (1) Changing the Inference Backend

  ##### For the general OCR pipeline, use the `onnxruntime` backend for all models:

  <details><summary>👉 1. Modify via Pipeline Configuration File (click to expand)</summary>

  ```yaml
  pipeline_name: OCR

  hpi_config:
    backend: onnxruntime

  ...
  ```

  </details>
  <details><summary>👉 2. CLI Parameter Method (click to expand)</summary>

  ```bash
  paddlex \
      --pipeline image_classification \
      --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
      --device gpu:0 \
      --use_hpip \
      --hpi_config '{"backend": "onnxruntime"}'
  ```

  </details>
  <details><summary>👉 3. Python API Parameter Method (click to expand)</summary>

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

  ##### For the image classification module, use the `onnxruntime` backend:

  <details><summary>👉 1. Modify via Pipeline Configuration File (click to expand)</summary>

  ```yaml
  # paddlex/configs/modules/image_classification/ResNet18.yaml
  ...
  Predict:
    ...
    hpi_config:
        backend: onnxruntime
    ...
  ...
  ```

  </details>
  <details><summary>👉 2. CLI Parameter Method (click to expand)</summary>

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
  <details><summary>👉 3. Python API Parameter Method (click to expand)</summary>

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

  ##### For the general OCR pipeline, use the `onnxruntime` backend for the `text_detection` module and the `tensorrt` backend for the `text_recognition` module:

  <details><summary>👉 1. Modify via Pipeline Configuration File (click to expand)</summary>

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
      hpi_config:
        backend: tensorrt
  ```

  </details>

#### (2) Modifying TensorRT Dynamic Shape Configuration

  ##### For the general image classification pipeline, modify dynamic shape configuration:

  <details><summary>👉 Click to expand</summary>

  ```yaml
    ...
    SubModules:
      ImageClassification:
        ...
        hpi_config:
          backend: tensorrt
          backend_config:
            dynamic_shapes:
              x:
                - [1, 3, 300, 300]
                - [4, 3, 300, 300]
                - [32, 3, 1200, 1200]
              ...
    ...
  ```

  </details>

  ##### For the image classification module, modify dynamic shape configuration:

  <details><summary>👉 Click to expand</summary>

  ```yaml
  ...
  Predict:
    ...
    hpi_config:
        backend: tensorrt
        backend_config:
          dynamic_shapes:
            x:
              - [1, 3, 300, 300]
              - [4, 3, 300, 300]
              - [32, 3, 1200, 1200]
    ...
  ...
  ```

  </details>

### 2.4 Enabling/Disabling the High-Performance Inference Plugin on Sub-pipelines/Submodules

High-performance inference supports enabling the high-performance inference plugin for only specific sub-pipelines/submodules by configuring `use_hpip` at the sub-pipeline or submodule level. For example:

##### In the general OCR pipeline, enable high-performance inference for the `text_detection` module, but not for the `text_recognition` module:

  <details><summary>👉 Click to expand</summary>

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
      use_hpip: True # This submodule uses high-performance inference
    TextLineOrientation:
      module_name: textline_orientation
      model_name: PP-LCNet_x0_25_textline_ori
      model_dir: null
      batch_size: 6
      # This submodule does not have a specific configuration; it defaults to the global configuration
      # (if neither the configuration file nor CLI/API parameters set it, high-performance inference will not be used)
    TextRecognition:
      module_name: text_recognition
      model_name: PP-OCRv4_mobile_rec
      model_dir: null
      batch_size: 6
      score_thresh: 0.0
      use_hpip: False # This submodule does not use high-performance inference
  ```

  </details>

**Note:**

1. When setting `use_hpip` in sub-pipelines or submodules, the configuration at the deepest level will take precedence.
2. **When enabling or disabling the high-performance inference plugin by modifying the pipeline configuration file, it is not recommended to also configure it using the CLI or Python API.** Setting `use_hpip` through the CLI or Python API is equivalent to modifying the top-level `use_hpip` in the configuration file.

### 2.5 Model Cache Description

The model cache is stored in the `.cache` directory under the model directory, including files such as `shape_range_info.pbtxt` and those starting with `trt_serialized` generated when using the `tensorrt` or `paddle` backends.

**After modifying TensorRT-related configurations, it is recommended to clear the cache to avoid the new configuration being overridden by the cache.**

When the `auto_paddle2onnx` option is enabled, an `inference.onnx` file may be automatically generated in the model directory.

### 2.6 Customizing the Model Inference Library

`ultra-infer` is the model inference library that the high-performance inference plugin depends on. It is maintained as a sub-project under the `PaddleX/libs/ultra-infer` directory. PaddleX provides a build script for `ultra-infer`, located at `PaddleX/libs/ultra-infer/scripts/linux/set_up_docker_and_build_py.sh`. The build script, by default, builds the GPU version of `ultra-infer` and integrates three inference backends: OpenVINO, TensorRT, and ONNX Runtime.

If you need to customize the build of `ultra-infer`, you can modify the following options in the build script according to your requirements:

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
            <td>The HTTP proxy used when downloading third-party libraries; default is empty.</td>
        </tr>
        <tr>
            <td>PYTHON_VERSION</td>
            <td>Python version, default is <code>3.10.0</code>.</td>
        </tr>
        <tr>
            <td>WITH_GPU</td>
            <td>Whether to enable GPU support, default is <code>ON</code>.</td>
        </tr>
        <tr>
            <td>ENABLE_ORT_BACKEND</td>
            <td>Whether to integrate the ONNX Runtime backend, default is <code>ON</code>.</td>
        </tr>
        <tr>
            <td>ENABLE_TRT_BACKEND</td>
            <td>Whether to integrate the TensorRT backend (GPU-only), default is <code>ON</code>.</td>
        </tr>
        <tr>
            <td>ENABLE_OPENVINO_BACKEND</td>
            <td>Whether to integrate the OpenVINO backend (CPU-only), default is <code>ON</code>.</td>
        </tr>
    </tbody>
</table>

Example:

```shell
# Build
cd PaddleX/libs/ultra-infer/scripts/linux
# export PYTHON_VERSION=...
# export WITH_GPU=...
# export ENABLE_ORT_BACKEND=...
# export ...
bash set_up_docker_and_build_py.sh

# Install
python -m pip install ../../python/dist/ultra_infer*.whl
```

## 3. Frequently Asked Questions

**1. Why does the inference speed not appear to improve noticeably before and after enabling the high-performance inference plugin?**

The high-performance inference plugin accelerates inference by intelligently selecting the backend.

For modules, due to model complexity or unsupported operators, some models may not be able to use acceleration backends (such as OpenVINO, TensorRT, etc.). In such cases, corresponding messages will be logged, and the fastest available backend known will be chosen, which may fall back to standard inference.

For model pipelines, the performance bottleneck may not lie in the inference stage.

You can use the [PaddleX benchmark](../module_usage/instructions/benchmark.md) tool to conduct actual speed tests for a more accurate performance evaluation.

**2. Does the high-performance inference functionality support all model pipelines and modules?**

The high-performance inference functionality supports all model pipelines and modules, but some models may not see an acceleration effect due to reasons mentioned in FAQ 1.

**3. Why does the installation of the high-performance inference plugin fail with a log message stating: “Currently, the CUDA version must be 11.x for GPU devices.”?**

The high-performance inference functionality currently supports only a limited set of environments. Please refer to the installation instructions. If installation fails, it may be that the current environment is not supported by the high-performance inference functionality. Note that CUDA 12.6 is already under support.

**4. Why does the program freeze during runtime or display some WARNING and ERROR messages after using the high-performance inference functionality? What should be done in such cases?**

When initializing the model, operations such as subgraph optimization may take longer and may generate some WARNING and ERROR messages. However, as long as the program does not exit automatically, it is recommended to wait patiently, as the program usually continues to run to completion.
