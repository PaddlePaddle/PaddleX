---
comments: true
---

# PaddleX Serving Guide

Serving is a common deployment strategy in real-world production environments. By encapsulating inference functions into services, clients can access these services via network requests to obtain inference results. PaddleX supports various solutions for serving pipelines.

Demonstration of PaddleX pipeline serving:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/serving_en.png" width="300"/>

To address different user needs, PaddleX offers multiple pipeline serving solutions:

- **Basic serving**: A simple and easy-to-use serving solution with low development costs.
- **High-stability serving**: Built on [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server). Compared to basic serving, this solution offers higher stability and allows users to adjust configurations to optimize performance.

**It is recommended to first use the basic serving solution for quick verification**, and then evaluate whether to try more complex solutions based on actual needs.

<b>Note</b>

- PaddleX serves pipelines rather than modules.

## 1. Basic Serving

### 1.1 Install the Serving Plugin

Execute the following command to install the serving plugin:

```bash
paddlex --install serving
```

### 1.2 Run the Server

Run the server via PaddleX CLI:

```bash
paddlex --serve --pipeline {pipeline name or path to pipeline config file} [{other commandline options}]
```

Take the general image classification pipeline as an example:

```bash
paddlex --serve --pipeline image_classification
```

You can see the following information:

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

`--pipeline` can be specified as the official pipeline name or the path to a local pipeline configuration file. PaddleX builds the pipeline and deploys it as a service. If you need to adjust configurations (such as model paths, batch size, device for deployment, etc.), please refer to the <b>"Model Application"</b> section in [General Image Classification Pipeline Tutorial](../pipeline_usage/tutorials/cv_pipelines/image_classification.en.md).

The command-line options related to serving are as follows:

<table>
<thead>
<tr>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>Pipeline name or path to the pipeline configuration file.</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>Device for pipeline deployment. By default the GPU will be used when it is available; otherwise, the CPU will be used.</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>Hostname or IP address the server binds to. Defaults to <code>0.0.0.0</code>.</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>Port number the server listens on. Defaults to <code>8080</code>.</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>If specified, enables the high-performance inference plugin.</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>High-performance inference configuration</td>
</tr>
</tbody>
</table>
</table>

In application scenarios where strict requirements are placed on service response time, the PaddleX high-performance inference plugin can be used to accelerate model inference and pre/post-processing, thereby reducing response time and increasing throughput.

To use the PaddleX high-performance inference plugin, please refer to the [PaddleX High-Performance Inference Guide](./high_performance_inference.en.md).

You can use the `--use_hpip` flag to enable the high-performance inference plugin. An example is as follows:

```bash
paddlex --serve --pipeline image_classification --use_hpip
```

### 1.3 Invoke the Service

The "Development Integration/Deployment" section in each pipeline’s tutorial provides API references and multi-language invocation examples for the service. You can find the tutorials for each pipeline [here](../pipeline_usage/pipeline_develop_guide.en.md).

## 2. High-Stability Serving

**Please note that the current high-stability serving solution only supports Linux systems.**

### 2.1 Download the High-Stability Serving SDK

Find the high-stability serving SDK corresponding to the pipeline in the table below and download it:

<details>
<summary>👉 Click to view</summary>
<table>
<thead>
<tr>
<th>Pipeline</th>
<th>SDK</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-ChatOCR-doc v3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_PP-ChatOCRv3-doc_sdk.tar.gz">paddlex_hps_PP-ChatOCRv3-doc_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General image classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_image_classification_sdk.tar.gz">paddlex_hps_image_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General object detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_object_detection_sdk.tar.gz">paddlex_hps_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General instance segmentation</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_instance_segmentation_sdk.tar.gz">paddlex_hps_instance_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General semantic segmentation</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_semantic_segmentation_sdk.tar.gz">paddlex_hps_semantic_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Image multi-label classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_image_multilabel_classification_sdk.tar.gz">paddlex_hps_image_multilabel_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General image recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_PP-ShiTuV2_sdk.tar.gz">paddlex_hps_PP-ShiTuV2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Pedestrian attribute recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_pedestrian_attribute_recognition_sdk.tar.gz">paddlex_hps_pedestrian_attribute_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Vehicle attribute recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_vehicle_attribute_recognition_sdk.tar.gz">paddlex_hps_vehicle_attribute_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Face recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_face_recognition_sdk.tar.gz">paddlex_hps_face_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Small object detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_small_object_detection_sdk.tar.gz">paddlex_hps_small_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Image anomaly detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_anomaly_detection_sdk.tar.gz">paddlex_hps_anomaly_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Human keypoint detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_human_keypoint_detection_sdk.tar.gz">paddlex_hps_human_keypoint_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Open vocabulary detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_open_vocabulary_detection_sdk.tar.gz">paddlex_hps_open_vocabulary_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Open vocabulary segmentation</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_open_vocabulary_segmentation_sdk.tar.gz">paddlex_hps_open_vocabulary_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Rotated object detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_rotated_object_detection_sdk.tar.gz">paddlex_hps_rotated_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>3D multi-modal fusion detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_3d_bev_detection_sdk.tar.gz">paddlex_hps_3d_bev_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General OCR</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_OCR_sdk.tar.gz">paddlex_hps_OCR_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General table recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_table_recognition_sdk.tar.gz">paddlex_hps_table_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General table recognition v2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_table_recognition_v2_sdk.tar.gz">paddlex_hps_table_recognition_v2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General layout parsing</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_layout_parsing_sdk.tar.gz">paddlex_hps_layout_parsing_sdk.tar.gz</a></td>
</tr>
<tr>
<td>PP-StructureV3</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_PP-StructureV3_sdk.tar.gz">paddlex_hps_PP-StructureV3_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Formula recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_formula_recognition_sdk.tar.gz">paddlex_hps_formula_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Seal text recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_seal_recognition_sdk.tar.gz">paddlex_hps_seal_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Document image preprocessing</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_doc_preprocessor_sdk.tar.gz">paddlex_hps_doc_preprocessor_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Time series forecasting</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_ts_forecast_sdk.tar.gz">paddlex_hps_ts_forecast_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Time series anomaly detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_ts_anomaly_detection_sdk.tar.gz">paddlex_hps_ts_anomaly_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Time series classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_ts_classification_sdk.tar.gz">paddlex_hps_ts_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Multilingual speech recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_multilingual_speech_recognition_sdk.tar.gz">paddlex_hps_multilingual_speech_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General video classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_video_classification_sdk.tar.gz">paddlex_hps_video_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General video detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_video_detection_sdk.tar.gz">paddlex_hps_video_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Document understanding</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.1/paddlex_hps_doc_understanding_sdk.tar.gz">paddlex_hps_doc_understanding_sdk.tar.gz</a></td>
</tr>
</tbody>
</table>
</details>

### 2.2 Adjust Configurations

The `server/pipeline_config.yaml` file of the the high-stability serving SDK is the pipeline configuration file. Users can modify this file to set the model directory to use, etc.

In addition, the PaddleX high-stability serving solution is built on NVIDIA Triton Inference Server, allowing users to modify the configuration files of Triton Inference Server.

In the `server/model_repo/{endpoint name}` directory of the high-stability serving SDK, you can find one or more `config*.pbtxt` files. If a `config_{device type}.pbtxt` file exists in the directory, please modify the configuration file corresponding to the desired device type. Otherwise, please modify `config.pbtxt`.

A common requirement is to adjust the number of execution instances. To achieve this, you need to modify the `instance_group` setting in the configuration file, using `count` to specify the number of instances placed on each device, `kind` to specify the device type, and `gpus` to specify the GPU IDs. An example is as follows:

- Place 4 instances on GPU 0:

    ```text
    instance_group [
    {
        count: 4
        kind: KIND_GPU
        gpus: [ 0 ]
    }
    ]
    ```

- Place 2 instances on GPU 1, and 1 instance each on GPUs 2 and 3:

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

For more configuration details, please refer to the [Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).

### 2.3 Run the Server

The machine used for deployment needs to have Docker Engine version 19.03 or higher installed.

First, pull the Docker image as needed:

- Image supporting deployment with NVIDIA GPU (the machine must have NVIDIA drivers that support CUDA 11.8 installed):

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.1-gpu
    ```

- CPU-only Image:

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.1-cpu
    ```

With the image prepared, navigate to the `server` directory and execute the following command to run the server:

```bash
docker run \
    -it \
    -e PADDLEX_HPS_DEVICE_TYPE={deployment device type} \
    -v "$(pwd)":/app \
    -w /app \
    --rm \
    --gpus all \
    --init \
    --network host \
    --shm-size 8g \
    {image name} \
    /bin/bash server.sh
```

- The deployment device type can be `cpu` or `gpu`, and the CPU-only image supports only `cpu`.
- If CPU deployment is required, there is no need to specify `--gpus`.
- If you need to enter the container for debugging, you can replace `/bin/bash server.sh` in the command with `/bin/bash`. Then execute `/bin/bash server.sh` inside the container.
- If you want the server to run in the background, you can replace `-it` in the command with `-d`. After the container starts, you can view the container logs with `docker logs -f {container ID}`.
- Add `-e PADDLEX_HPS_USE_HPIP=1` to use the PaddleX high-performance inference plugin to accelerate the pipeline inference process. Please refer to the [PaddleX High-Performance Inference Guide](./high_performance_inference.en.md) for more information.

You may observe output similar to the following:

```text
I1216 11:37:21.601943 35 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I1216 11:37:21.602333 35 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I1216 11:37:21.643494 35 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

### 2.4 Invoke the Service

Users can call the pipeline service through the Python client provided by the SDK or by manually constructing HTTP requests (with no restriction on client programming languages). 


The services deployed using the high-stability serving solution offer the primary operations that match those of the basic serving solution. For each primary operation, the endpoint names and the request and response data fields are consistent with the basic serving solution. Please refer to the "Development Integration/Deployment" section in the tutorials for each pipeline. The tutorials for each pipeline can be found [here](../pipeline_usage/pipeline_develop_guide.en.md).


#### 2.4.1 Use Python Client

Navigate to the `client` directory of the high-stability serving SDK, and run the following command to install dependencies:

```bash
# It is recommended to install in a virtual environment
python -m pip install -r requirements.txt
python -m pip install paddlex_hps_client-*.whl
```

The Python client currently supports Python versions 3.8 to 3.12.

The `client.py` script in the `client` directory contains examples of how to call the service and provides a command-line interface.

#### 2.4.2 Manually Construct HTTP Requests

The following method demonstrates how to call the service using the HTTP interface in scenarios where the Python client is not applicable.

First, you need to manually construct the HTTP request body. The request body must be in JSON format and contains the following fields:

- `inputs`: Input tensor information. The input tensor name `name` is uniformly set to `input`, the shape is `[1, 1]`, and the data type `datatype` is `BYTES`. The  tensor data `data` contains a single JSON string, and the content of this JSON should follow the pipeline-specific format (consistent with the basic serving solution).
- `outputs`: Output tensor information. The output tensor name `name` is uniformly set to `output`.

Taking the general OCR pipeline as an example, the constructed request body is as follows:

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

Send the constructed request body to the corresponding HTTP inference endpoint of the service. By default, the service listens on HTTP port `8000`, and the inference request URL follows the format `http://{hostname}:8000/v2/models/{endpoint name}/infer`.

Using the general OCR pipeline as an example, the following is a `curl` command to send the request:

```bash
# Assuming `REQUEST_JSON` is the request body constructed in the previous step
curl -s -X POST http://localhost:8000/v2/models/ocr/infer \
    -H 'Content-Type: application/json' \
    -d "${REQUEST_JSON}"
```

Finally, the response from the service needs to be parsed. The raw response body has the following structure:

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

`outputs[0].data[0]` is a JSON string. The internal fields follow the same format as the response body in the basic serving solution. For detailed parsing rules, please refer to the usage guide for each specific pipeline.
