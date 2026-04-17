---
comments: true
---

# PaddleX 服务化部署指南

服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。

PaddleX 产线服务化部署示意图：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/serving.png" width="300"/>

针对用户的不同需求，PaddleX 提供多种产线服务化部署方案：

- **基础服务化部署**：简单易用的服务化部署方案，开发成本低。
- **高稳定性服务化部署**：基于 [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server) 打造。与基础服务化部署相比，该方案提供更高的稳定性，并允许用户调整配置以优化性能。

**建议首先使用基础服务化部署方案进行快速验证**，然后根据实际需要，评估是否尝试更复杂的方案。

**注意**

- PaddleX 对产线而不是模块进行服务化部署。

## 1. 基础服务化部署

### 1.1 安装服务化部署插件

执行如下命令，安装服务化部署插件：

```bash
paddlex --install serving
```

### 1.2 运行服务器

通过 PaddleX CLI 运行服务器：

```bash
paddlex --serve --pipeline {产线名称或产线配置文件路径} [{其他命令行选项}]
```

以通用图像分类产线为例：

```bash
paddlex --serve --pipeline image_classification
```

可以看到类似以下展示的信息：

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

`--pipeline` 可指定为官方产线名称或本地产线配置文件路径。PaddleX 以此构建产线并部署为服务。如需调整配置（如模型路径、batch size、部署设备等），请参考 [通用图像分类产线使用教程](../pipeline_usage/tutorials/cv_pipelines/image_classification.md) 中的 <b>”模型应用“</b> 部分。

与服务化部署相关的命令行选项如下：

<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>产线名称或产线配置文件路径。</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>产线部署设备。默认情况下，当 GPU 可用时，将使用 GPU；否则使用 CPU。</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>服务器绑定的主机名或 IP 地址。默认为 <code>0.0.0.0</code>。</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>服务器监听的端口号。默认为 <code>8080</code>。</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>如果指定，则启用高性能推理插件。</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>高性能推理配置。</td>
</tr>
</tbody>
</table>
</table>

在对于服务响应时间要求较严格的应用场景中，可以使用 PaddleX 高性能推理插件对模型推理及前后处理进行加速，从而降低响应时间、提升吞吐量。

使用 PaddleX 高性能推理插件，请参考 [PaddleX 高性能推理指南](./high_performance_inference.md) 。

可以通过指定 `--use_hpip` 以使用高性能推理插件。示例如下：

```bash
paddlex --serve --pipeline image_classification --use_hpip
```

### 1.3 调用服务

各产线使用教程中的 <b>“开发集成/部署”</b> 部分提供了服务的 API 参考与多语言调用示例。在 [此处](../pipeline_usage/pipeline_develop_guide.md) 可以找到各产线的使用教程。

## 2. 高稳定性服务化部署

**请注意，当前高稳定性服务化部署方案仅支持 Linux 系统。**

### 2.1 下载高稳定性服务化部署 SDK

在下表中找到产线对应的高稳定性服务化部署 SDK 并下载：

<details>
<summary>👉 点击查看</summary>
<table>
<thead>
<tr>
<th>产线</th>
<th>SDK</th>
</tr>
</thead>
<tbody>
<tr>
<td>文档场景信息抽取 v3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_PP-ChatOCRv3-doc_sdk.tar.gz">paddlex_hps_PP-ChatOCRv3-doc_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用图像分类</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_image_classification_sdk.tar.gz">paddlex_hps_image_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用目标检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_object_detection_sdk.tar.gz">paddlex_hps_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用实例分割</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_instance_segmentation_sdk.tar.gz">paddlex_hps_instance_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用语义分割</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_semantic_segmentation_sdk.tar.gz">paddlex_hps_semantic_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用图像多标签分类</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_image_multilabel_classification_sdk.tar.gz">paddlex_hps_image_multilabel_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用图像识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_PP-ShiTuV2_sdk.tar.gz">paddlex_hps_PP-ShiTuV2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>行人属性识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_pedestrian_attribute_recognition_sdk.tar.gz">paddlex_hps_pedestrian_attribute_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>车辆属性识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_vehicle_attribute_recognition_sdk.tar.gz">paddlex_hps_vehicle_attribute_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>人脸识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_face_recognition_sdk.tar.gz">paddlex_hps_face_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>小目标检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_small_object_detection_sdk.tar.gz">paddlex_hps_small_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>图像异常检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_anomaly_detection_sdk.tar.gz">paddlex_hps_anomaly_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>人体关键点检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_human_keypoint_detection_sdk.tar.gz">paddlex_hps_human_keypoint_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>开放词汇检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_open_vocabulary_detection_sdk.tar.gz">paddlex_hps_open_vocabulary_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>开放词汇分割</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_open_vocabulary_segmentation_sdk.tar.gz">paddlex_hps_open_vocabulary_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>旋转目标检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_rotated_object_detection_sdk.tar.gz">paddlex_hps_rotated_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>3D 多模态融合检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_3d_bev_detection_sdk.tar.gz">paddlex_hps_3d_bev_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用 OCR</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_OCR_sdk.tar.gz">paddlex_hps_OCR_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用表格识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_table_recognition_sdk.tar.gz">paddlex_hps_table_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用表格识别 v2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_table_recognition_v2_sdk.tar.gz">paddlex_hps_table_recognition_v2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用版面解析</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_layout_parsing_sdk.tar.gz">paddlex_hps_layout_parsing_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用版面解析 v3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_PP-StructureV3_sdk.tar.gz">paddlex_hps_PP-StructureV3_sdk.tar.gz</a></td>
</tr>
<tr>
<td>公式识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_formula_recognition_sdk.tar.gz">paddlex_hps_formula_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>印章文本识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_seal_recognition_sdk.tar.gz">paddlex_hps_seal_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>文档图像预处理</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_doc_preprocessor_sdk.tar.gz">paddlex_hps_doc_preprocessor_sdk.tar.gz</a></td>
</tr>
<tr>
<td>时序预测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_ts_forecast_sdk.tar.gz">paddlex_hps_ts_forecast_sdk.tar.gz</a></td>
</tr>
<tr>
<td>时序异常检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_ts_anomaly_detection_sdk.tar.gz">paddlex_hps_ts_anomaly_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>时序分类</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_ts_classification_sdk.tar.gz">paddlex_hps_ts_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>多语种语音识别</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_multilingual_speech_recognition_sdk.tar.gz">paddlex_hps_multilingual_speech_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用视频分类</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_video_classification_sdk.tar.gz">paddlex_hps_video_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>通用视频检测</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_video_detection_sdk.tar.gz">paddlex_hps_video_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>文档理解</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_doc_understanding_sdk.tar.gz">paddlex_hps_doc_understanding_sdk.tar.gz</a></td>
</tr>
<tr>
<td>PaddleOCR-VL</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_PaddleOCR-VL_sdk.tar.gz">paddlex_hps_PaddleOCR-VL_sdk.tar.gz</a></td>
</tr>
<tr>
<td>PaddleOCR-VL-1.5</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.5/paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz">paddlex_hps_PaddleOCR-VL-1.5_sdk.tar.gz</a></td>
</tr>
</tbody>
</table>
</details>

如需手动打包可参考 [`hps` 项目文档](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/hps/README.md#2-产线物料打包)。

### 2.2 调整配置

高稳定性服务化部署 SDK 的 `server/pipeline_config.yaml` 文件为产线配置文件。用户可以修改该文件以设置要使用的模型目录等。

此外，PaddleX 高稳定性服务化部署方案基于 NVIDIA Triton Inference Server 打造，支持用户修改 Triton Inference Server 的配置文件。

在高稳定性服务化部署 SDK 的 `server/model_repo/{端点名称}` 目录中，可以找到一个或多个 `config*.pbtxt` 文件。如果目录中存在 `config_{设备类型}.pbtxt` 文件，请修改期望使用的设备类型对应的配置文件；否则，请修改 `config.pbtxt`。

一个常见的需求是调整执行实例数量。为了实现这一点，需要修改配置文件中的 `instance_group` 配置，使用 `count` 指定每一设备上放置的实例数量，使用 `kind` 指定设备类型，使用 `gpus` 指定 GPU 编号。示例如下：

- 在 GPU 0 上放置 4 个实例：

    ```text
    instance_group [
    {
        count: 4
        kind: KIND_GPU
        gpus: [ 0 ]
    }
    ]
    ```

- 在 GPU 1 上放置 2 个实例，在 GPU 2 和 3 上分别放置 1 个实例：

    ```text
    instance_group [
    {
        count: 2
        kind: KIND_GPU
        gpus: [ 1 ]
    },
    {
        count: 1
        kind: KIND_GPU
        gpus: [ 2, 3 ]
    }
    ]
    ```

关于更多配置细节，请参阅 [Triton Inference Server 文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)。

### 2.3 运行服务器

用于部署的机器上需要安装有 19.03 或更高版本的 Docker Engine。

首先，根据需要拉取 Docker 镜像：

- 支持使用 NVIDIA GPU 部署的镜像（机器上需要安装有支持 CUDA 11.8 的 NVIDIA 驱动）：

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.5-gpu
    ```

- CPU-only 镜像：

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.5-cpu
    ```

如需自定义构建镜像可参考 [`hps` 项目文档](https://github.com/PaddlePaddle/PaddleX/blob/develop/deploy/hps/README.md#1-镜像构建)。

准备好镜像后，切换到 `server` 目录，执行如下命令运行服务器：

```bash
docker run \
    -it \
    -e PADDLEX_HPS_DEVICE_TYPE={部署设备类型} \
    -v "$(pwd)":/app \
    -w /app \
    --rm \
    --gpus all \
    --init \
    --network host \
    --shm-size 8g \
    {镜像名称} \
    /bin/bash server.sh
```

- 部署设备类型可以为 `cpu` 或 `gpu`，CPU-only 镜像仅支持 `cpu`。
- 如果希望使用 CPU 部署，则不需要指定 `--gpus`。
- 如果需要进入容器内部调试，可以将命令中的 `/bin/bash server.sh` 替换为 `/bin/bash`，然后在容器中执行 `/bin/bash server.sh`。
- 如果希望服务器在后台运行，可以将命令中的 `-it` 替换为 `-d`。容器启动后，可通过 `docker logs -f {容器 ID}` 查看容器日志。
- 在命令中添加 `-e PADDLEX_HPS_USE_HPIP=1` 可以使用 PaddleX 高性能推理插件加速产线推理过程。请参考 [PaddleX 高性能推理指南](./high_performance_inference.md) 获取更多信息。

可观察到类似下面的输出信息：

```text
I1216 11:37:21.601943 35 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I1216 11:37:21.602333 35 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I1216 11:37:21.643494 35 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

### 2.4 调用服务

用户可以通过 SDK 中的 Python 客户端调用产线服务，或者手动构造 HTTP 请求（对客户端语言无限制）。

使用高稳定性服务化部署方案部署的服务，提供与基础服务化部署方案相匹配的主要操作。对于每个主要操作，端点名称以及请求和响应的数据字段都与基础服务化部署方案保持一致。请参阅各产线使用教程中的 <b>“开发集成/部署”</b> 部分。在 [此处](../pipeline_usage/pipeline_develop_guide.md) 可以找到各产线的使用教程。

#### 2.4.1 使用 Python 客户端

切换到高稳定性服务化部署 SDK 的 `client` 目录，执行如下命令安装依赖：

```bash
# 建议在虚拟环境中安装
python -m pip install -r requirements.txt
python -m pip install paddlex_hps_client-*.whl
```

Python 客户端目前支持的 Python 版本为 3.8 至 3.12。

`client` 目录的 `client.py` 脚本包含服务的调用示例，并提供命令行接口。

#### 2.4.2 手动构造 HTTP 请求

以下方式手工构造 HTTP 请求体并调用，适用于 Python 客户端不适用的情形。

首先，需要构造请求体。请求体为 JSON 格式，包含以下字段：

- `inputs`：输入张量信息。输入张量名称 `name` 统一为 `input`，张量形状 `shape` 为 `[1, 1]`，数据类型 `datatype` 为 `BYTES`。张量数据 `data` 包含一个 JSON 字符串，JSON 的内容需对应不同产线字段（与基础服务化部署一致）。
- `outputs`：输出张量信息。输出张量名称 `name` 统一为`output`。

以通用 OCR 产线为例，构造的请求体内容示例如下：

```JSON
{
  "inputs": [
    {
      "name": "input",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": [
        "{\"file\":\"https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png\",\"visualize\":false}"
      ]
    }
  ],
  "outputs": [
    {
      "name": "output"
    }
  ]
}
```

将构造好的请求体通过发送到服务对应的 HTTP 推理端点。服务默认监听的 HTTP 端口为 `8000`，推理请求的 URL 格式为 `http://{主机名}:8000/v2/models/{端点名称}/infer`。

以通用 OCR 产线为例，如下是通过 `curl` 向发送请求的例子：

```bash
# 假设 `REQUEST_JSON` 为上一步骤中构造的请求体
curl -s -X POST http://localhost:8000/v2/models/ocr/infer \
    -H 'Content-Type: application/json' \
    -d "${REQUEST_JSON}"
```

最后，需要解析服务的响应。响应体的原始结构如下：

```json
{
  "outputs": [
    {
      "name": "output",
      "data": [
        "{\"errorCode\": 0, \"result\": {\"ocrResults\": [...]}}"
      ]
    }
  ]
}
```

其中 `outputs[0].data[0]` 是一个 JSON 字符串，其中的字段与基础服务化部署方案中的响应体保持一致，具体解析规则可以查看各产线使用教程。
