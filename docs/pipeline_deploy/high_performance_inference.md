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
  - [2.2 二次开发高性能推理插件](#2.2-二次开发高性能推理插件)
- [3. 支持使用高性能推理插件的产线与模型](#3.-支持使用高性能推理插件的产线与模型)

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
    <td rowspan="7">x86-64</td>
    <td rowspan="7">Linux</td>
    <td rowspan="4">CPU</td>
  </tr>
  <tr>
    <td>3.8</td>
  </tr>
  <tr>
    <td>3.9</td>
  </tr>
  <tr>
    <td>3.10</td>
  </tr>
  <tr>
    <td rowspan="3">GPU&nbsp;（CUDA&nbsp;11.8&nbsp;+&nbsp;cuDNN&nbsp;8.6）</td>
    <td>3.8</td>
  </tr>
  <tr>
    <td>3.9</td>
  </tr>
  <tr>
    <td>3.10</td>
  </tr>
</table>

### 1.2 启用高性能推理插件

对于 PaddleX CLI，指定 `--use_hpip`，即可启用高性能推理插件。以通用图像分类产线为例：

```bash
paddlex \
    --pipeline image_classification \
    --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg \
    --device gpu:0 \
    --use_hpip
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

PaddleX 结合模型信息与运行环境信息为每个模型提供默认的高性能推理配置，其中包括推理后端和推理后端的配置。例如 ResNet18 模型，CPU 设备默认的推理后端为 onnx_runtime，GPU 设备默认的推理后端为 tensorrt FP16。
这些默认配置经过精心准备，以便在数个常见场景中可用，且能够取得较优的性能。因此，通常用户可能并不用关心如何这些配置的具体细节。
然而，由于实际部署环境与需求的多样性，使用默认配置可能无法在特定场景获取理想的性能，甚至可能出现推理失败的情况。对于默认配置无法满足要求的情形，用户可以手动调整配置。以下列举两种常见的情形：

- 更换推理后端：

    对于模型产线，通过在产线 yaml 中增加 `hpi_params` 字段，即可更换推理后端，以通用图像分类产线的 `image_classification.yaml` 为例：

    ```yaml
      ...
      SubModules:
        ImageClassification:
          ...
          hpi_params:
            config:
              selected_backends:
                cpu: openvino # 可选：paddle_infer, openvino, onnx_runtime
                gpu: paddle_infer # 可选：paddle_infer, onnx_runtime, tensorrt
              backend_config:
                # Paddle Inference 后端配置
                paddle_infer:
                  enable_trt: True # 可选：True, False
                  trt_precision: FP16 # 当 enable_trt 为 True 时，可选：FP32, FP16
                # TensorRT 后端配置
                tensorrt:
                  precision: FP32 # 可选：FP32, FP16
      ...
    ```

    对于单功能模块，通过传入 `hpi_params` 参数，即可更换推理后端，以图像分类模块为例：

    ```python
    from paddlex import create_model

    model = create_model(
        "ResNet18",
        device="gpu",
        use_hpip=True,
        hpi_params={
            "config": {
                "selected_backends": {"cpu": "openvino", "gpu": "paddle_infer"},
                "backend_config": {"paddle_infer": {"enable_trt": True, "trt_precision": "FP16"}, "tensorrt": {"precision": "FP32"}}
            }
        }
    )

    output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg")
    ```

    目前所有可选的推理后端如下：

    * `paddle_infer`：Paddle Inference 推理引擎。支持 CPU 和 GPU。相比 PaddleX 快速推理，高性能推理插件支持以集成 TensorRT 子图的方式提升模型的 GPU 推理性能。
    * `openvino`：[OpenVINO](https://github.com/openvinotoolkit/openvino)，Intel 提供的深度学习推理工具，优化了多种 Intel 硬件上的模型推理性能。仅支持 CPU。高性能推理插件自动将模型转换为 ONNX 格式后用该引擎推理。
    * `onnx_runtime`：[ONNX Runtime](https://onnxruntime.ai/)，跨平台、高性能的推理引擎。支持 CPU 和 GPU。高性能推理插件自动将模型转换为 ONNX 格式后用该引擎推理。
    * `tensorrt`：[TensorRT](https://developer.nvidia.com/tensorrt)，NVIDIA 提供的高性能深度学习推理库，针对 NVIDIA GPU 进行优化以提升速度。仅支持 GPU。高性能推理插件自动将模型转换为 ONNX 格式后用该引擎推理。

- 修改 Paddle Inference 或 TensorRT 的动态形状配置：

  动态形状是 TensorRT 延迟指定部分或全部张量维度直到运行时的能力。当默认的动态形状配置无法满足需求（例如，模型可能需要范围外的输入形状），用户需要修改相应的配置：

  对于模型产线，在产线 yaml 中的 `hpi_params` 字段中新增`trt_dynamic_shapes` 或 `dynamic_shapes` 字段，以通用图像分类产线的 `image_classification.yaml` 为例：

    ```yaml
      ...
      SubModules:
        ImageClassification:
          ...
          hpi_params:
            config:
              selected_backends:
                cpu: openvino
                gpu: paddle_infer
              backend_config:
                # Paddle Inference 后端配置
                paddle_infer:
                  enable_trt: True
                  trt_precision: FP16
                  trt_dynamic_shapes:
                    x:
                      - [1, 3, 300, 300]
                      - [4, 3, 300, 300]
                      - [32, 3, 1200, 1200]
                # TensorRT 后端配置
                tensorrt:
                  precision: FP32
                  dynamic_shapes:
                    x:
                      - [1, 3, 300, 300]
                      - [4, 3, 300, 300]
                      - [32, 3, 1200, 1200]
                  ...
      ...
    ```

    对于单功能模块，在 `hpi_params` 参数中新增 `trt_dynamic_shapes` 或 `dynamic_shapes` 字段，以图像分类模块为例：

    ```python
    from paddlex import create_model

    model = create_model(
        "ResNet18",
        device="gpu",
        use_hpip=True,
        hpi_params={
            "config": {
                "selected_backends": {"cpu": "openvino", "gpu": "paddle_infer"},
                "backend_config": {
                    # Paddle Inference 后端配置
                    "paddle_infer": {
                        "enable_trt": True,
                        "trt_precision": "FP16",
                        "trt_dynamic_shapes": {
                            "x": [
                                [1, 3, 300, 300],
                                [4, 3, 300, 300],
                                [32, 3, 1200, 1200]
                            ]
                        }
                    },
                    # TensorRT 后端配置
                    "tensorrt": {
                        "precision": "FP32",
                        "dynamic_shapes": {
                            "x": [
                                [1, 3, 300, 300],
                                [4, 3, 300, 300],
                                [32, 3, 1200, 1200]
                            ]
                        }
                    }
                }
            }
        }
    )

    output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg"
    ```

    在 `trt_dynamic_shapes` 或 `dynamic_shapes` 中，需要为每一个输入张量指定动态形状，格式为：`{输入张量名称}: [{最小形状}, [{最优形状}], [{最大形状}]]`。有关最小形状、最优形状以及最大形状的相关介绍及更多细节，请参考 [TensorRT 官方文档](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)。

    在完成修改后，请删除模型目录中的缓存文件（`shape_range_info.pbtxt` 与 `trt_serialized` 开头的文件）。

### 2.2 二次开发高性能推理插件

我们已经提供了完善的配置，通常情况下不建议进行二次开发。如果有以下需求，确实需要进行二次开发，请务必在充分评估后再进行。如以下场景：

- 自定义数据预处理或后处理逻辑。
- 实现特定算子的优化。
- 支持特殊的输入/输出格式。
- 集成第三方加速库。
- ......

二次开发高性能推理插件流程如下：

#### a. 按需修改 `ultra-infer` 代码

`ultra-infer`，是高性能推理功能的底层依赖，包含前后处理加速和多后端推理。位于 `libs` 目录下。

#### b. 安装 `ultra-infer`

对 `ultra-infer` 修改完成后，通过如下方式安装 `ultra-infer`。

`ultra-infer` 需要编译whl包，编译脚本位于 `PaddleX/libs/ultra-infer/scripts/linux/set_up_docker_and_build_py.sh` ，编译默认编译GPU版本和包含 `Paddle Inference`、`OpenVINO`、`TensorRT`、`ONNX Runtime` 四种推理后端的 `ultra-infer`。

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
| 选项 | 说明 |
|:------------------------|:------------------------------------|
| http_proxy             | 在下载三方库时使用具体的http代理，默认空 |
| PYTHON_VERSION | Python版本，默认 `3.10.0` |
| WITH_GPU | 是否编译支持Nvidia-GPU，默认 `ON` |
| ENABLE_ORT_BACKEND      | 是否编译集成ONNX Runtime后端，默认 `ON` |
| ENABLE_PADDLE_BACKEND   | 是否编译集成Paddle Inference后端，默认 `ON` |
| ENABLE_TRT_BACKEND   | 是否编译集成TensorRT后端，默认 `ON` |
| ENABLE_OPENVINO_BACKEND | 是否编译集成OpenVINO后端(仅支持CPU)，默认 `ON` |
| ENABLE_VISION           | 是否编译集成视觉模型的部署模块，默认 `ON` |
| ENABLE_TEXT             | 是否编译集成文本NLP模型的部署模块，默认 `ON` |

## 3. 支持使用高性能推理插件的产线与模型

<table>
  <tr>
    <th>模型产线</th>
    <th>单功能模块</th>
    <th>支持数量/模型总数</th>
    <th>不支持模型</th>
  </tr>

  <tr>
    <td rowspan="2">通用OCR</td>
    <td>文本检测</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本识别</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="7">文档场景信息抽取v3</td>
    <td>表格结构识别</td>
    <td><b>2</b> / 4 </td>
    <td>
      <details>
        <summary>查看详情</summary>
        SLANeXt_wired</br>
        SLANeXt_wireless</br>
      </details>
    </td>
  </tr>

  <tr>
    <td>版面区域检测</td>
    <td><b>11</b> / 11 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>文本检测</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本识别</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>印章文本检测</td>
    <td><b>2</b> / 2 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本图像矫正</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文档图像方向分类</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="4">通用表格识别</td>
    <td>版面区域检测</td>
    <td><b>11</b> / 11 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>表格结构识别</td>
    <td><b>2</b> / 4 </td>
    <td>
      <details>
        <summary>查看详情</summary>
        SLANeXt_wired</br>
        SLANeXt_wireless</br>
      </details>
    </td>
  </tr>

  <tr>
    <td>文本检测</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本识别</td>
    <td><b>4</b> / 4 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>通用目标检测</td>
    <td>目标检测</td>
    <td><b>34</b> / 41</td>
    <td>
      <details>
        <summary>查看详情</summary>
        FasterRCNN-Swin-Tiny-FPN<br>
        CenterNet-DLA-34<br>
        CenterNet-ResNet50<br>
        Co-DINO-R50<br>
        Co-DINO-Swin-L<br>
        Co-Deformable-DETR-R50<br>
        Co-Deformable-DETR-Swin-T<br>
      </details>
    </td>
  </tr>

  <tr>
    <td>通用实例分割</td>
    <td>实例分割</td>
    <td><b>12</b> / 15</td>
    <td>
      <details>
        <summary>查看详情</summary>
        Mask-RT-DETR-S</br>
        PP-YOLOE_seg-S</br>
        SOLOv2
      </details>
    </td>
  </tr>

  <tr>
    <td>通用图像分类</td>
    <td>图像分类</td>
    <td><b>80</b> / 80 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>通用语义分割</td>
    <td>语义分割</td>
    <td><b>20</b> / 20 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>时序预测</td>
    <td>时序预测</td>
    <td><b>7</b> / 7 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>时序异常检测</td>
    <td>时序异常预测</td>
    <td><b>4</b> / 5</td>
    <td>
      <details>
        <summary>查看详情</summary>
        TimesNet_ad</br>
      </details>
    </td>
  </tr>

  <tr>
    <td>时序分类</td>
    <td>时序分类</td>
    <td><b>1</b> / 1 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>小目标检测</td>
    <td>小目标检测</td>
    <td><b>3</b> / 3 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>图像多标签分类</td>
    <td>图像多标签分类</td>
    <td><b>6</b> / 6 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>图像异常检测</td>
    <td>无监督异常检测</td>
    <td><b>1</b> / 1 </td>
    <td>无</td>
  </tr>

  <tr>
    <td rowspan="8">通用版面解析</td>
    <td>表格结构识别</td>
    <td><b>2</b> / 4 </td>
    <td>
      <details>
        <summary>查看详情</summary>
        SLANeXt_wired</br>
        SLANeXt_wireless</br>
      </details>
    </td>
  </tr>

  <tr>
    <td>版面区域检测</td>
    <td><b>11</b> / 11 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>文本检测</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本识别</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>公式识别</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>印章文本检测</td>
    <td><b>2</b> / 2 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本图像矫正</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文档图像方向分类</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="2">公式识别</td>
    <td>版面区域检测</td>
    <td><b>11</b> / 11 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>公式识别</td>
    <td><b>1</b> / 4 </td>
    <td>
      <details>
        <summary>查看详情</summary>
        UnimerNet</br>
        PP-FormulaNet-L</br>
        PP-FormulaNet-S</br>
      </details>
    </td>
  </tr>

  <tr>
    <td rowspan="3">印章文本识别</td>
    <td>版面区域检测</td>
    <td><b>11</b> / 11 </td>
    <td>无</td>
  </tr>

  <tr>
    <td>印章文本检测</td>
    <td><b>2</b> / 2 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>文本识别</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="2">通用图像识别</td>
    <td>主体检测</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>图像特征</td>
    <td><b>3</b> / 3 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="2">行人属性识别</td>
    <td>行人检测</td>
    <td><b>2</b> / 2 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>行人属性识别</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="2">车辆属性识别</td>
    <td>车辆检测</td>
    <td><b>2</b> / 2 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>车辆属性识别</td>
    <td><b>1</b> / 1 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td rowspan="2">人脸识别</td>
    <td>人脸检测</td>
    <td><b>4</b> / 4 </td>
    <td>无 </td>
  </tr>

  <tr>
    <td>人脸特征</td>
    <td><b>2</b> / 2 </td>
    <td>无 </td>
  </tr>

</table>


<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/ultra_infer_python-1.0.0.3.0.0rc0-cp38-cp38-linux_x86_64.whl"></a>
<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/ultra_infer_python-1.0.0.3.0.0rc0-cp39-cp39-linux_x86_64.whl"></a>
<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/ultra_infer_python-1.0.0.3.0.0rc0-cp310-cp310-linux_x86_64.whl"></a>

<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/ultra_infer_gpu_python-1.0.0.3.0.0rc0-cp38-cp38-linux_x86_64.whl"></a>
<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/ultra_infer_gpu_python-1.0.0.3.0.0rc0-cp39-cp39-linux_x86_64.whl"></a>
<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/ultra_infer_gpu_python-1.0.0.3.0.0rc0-cp310-cp310-linux_x86_64.whl"></a>

<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/3.0.0rc0/paddlex_hpi-3.0.0rc0-py3-none-any.whl"></a>
