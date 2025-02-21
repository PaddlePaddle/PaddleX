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
<td>Device for pipeline deployment. Defaults to <code>cpu</code> (if GPU is unavailable) or <code>gpu</code> (if GPU is available).</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>Hostname or IP address the server binds to. Defaults to `0.0.0.0`.</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>Port number the server listens on. Defaults to `8080`.</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>If specified, enables the high-performance inference plugin.</td>
</tr>
</tbody>
</table>
</table>

In application scenarios where strict requirements are placed on service response time, the PaddleX high-performance inference plugin can be used to accelerate model inference and pre/post-processing, thereby reducing response time and increasing throughput.

To use the PaddleX high-performance inference plugin, please refer to the [PaddleX High-Performance Inference Guide](./high_performance_inference.en.md) for instructions on installing the high-performance inference plugin, obtaining a serial number, and completing the activation process. Additionally, note that not all pipelines, models, and environments support the use of the high-performance inference plugin. For detailed information on supported pipelines and models, please refer to the section on supported pipelines and models for high-performance inference plugins.

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
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_PP-ChatOCRv3-doc_sdk.tar.gz">paddlex_hps_PP-ChatOCRv3-doc_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General image classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_image_classification_sdk.tar.gz">paddlex_hps_image_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General object detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_object_detection_sdk.tar.gz">paddlex_hps_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General instance segmentation</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_instance_segmentation_sdk.tar.gz">paddlex_hps_instance_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General semantic segmentation</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_semantic_segmentation_sdk.tar.gz">paddlex_hps_semantic_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Image multi-label classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_image_multilabel_classification_sdk.tar.gz">paddlex_hps_image_multilabel_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General image recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_PP-ShiTuV2_sdk.tar.gz">paddlex_hps_PP-ShiTuV2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Pedestrian attribute recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_pedestrian_attribute_recognition_sdk.tar.gz">paddlex_hps_pedestrian_attribute_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Vehicle attribute recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_vehicle_attribute_recognition_sdk.tar.gz">paddlex_hps_vehicle_attribute_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Face recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_face_recognition_sdk.tar.gz">paddlex_hps_face_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Small object detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_small_object_detection_sdk.tar.gz">paddlex_hps_small_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Image anomaly detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_anomaly_detection_sdk.tar.gz">paddlex_hps_anomaly_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Human keypoint detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_human_keypoint_detection_sdk.tar.gz">paddlex_hps_human_keypoint_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Open vocabulary detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_open_vocabulary_detection_sdk.tar.gz">paddlex_hps_open_vocabulary_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Open vocabulary segmentation</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_open_vocabulary_segmentation_sdk.tar.gz">paddlex_hps_open_vocabulary_segmentation_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Rotated object detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_rotated_object_detection_sdk.tar.gz">paddlex_hps_rotated_object_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>3D multi-modal fusion detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_3d_bev_detection_sdk.tar.gz">paddlex_hps_3d_bev_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General OCR</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_OCR_sdk.tar.gz">paddlex_hps_OCR_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General table recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_table_recognition_sdk.tar.gz">paddlex_hps_table_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General table recognition v2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_table_recognition_v2_sdk.tar.gz">paddlex_hps_table_recognition_v2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General layout parsing</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_layout_parsing_sdk.tar.gz">paddlex_hps_layout_parsing_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General layout parsing v2</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_layout_parsing_v2_sdk.tar.gz">paddlex_hps_layout_parsing_v2_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Formula recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_formula_recognition_sdk.tar.gz">paddlex_hps_formula_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Seal text recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_seal_recognition_sdk.tar.gz">paddlex_hps_seal_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Document image preprocessing</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_doc_preprocessor_sdk.tar.gz">paddlex_hps_doc_preprocessor_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Time series forecasting</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_ts_forecast_sdk.tar.gz">paddlex_hps_ts_forecast_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Time series anomaly detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_ts_anomaly_detection_sdk.tar.gz">paddlex_hps_ts_anomaly_detection_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Time series classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_ts_classification_sdk.tar.gz">paddlex_hps_ts_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>Multilingual speech recognition</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_multilingual_speech_recognition_sdk.tar.gz">paddlex_hps_multilingual_speech_recognition_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General video classification</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_video_classification_sdk.tar.gz">paddlex_hps_video_classification_sdk.tar.gz</a></td>
</tr>
<tr>
<td>General video detection</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hps/public/sdks/v3.0.0rc0/paddlex_hps_video_detection_sdk.tar.gz">paddlex_hps_video_detection_sdk.tar.gz</a></td>
</tr>
</tbody>
</table>
</details>

### 2.2 Obtain the Serial Number

Using the high-stability serving solution requires obtaining a serial number and completing activation on the machine used for deployment. The method to obtain the serial number is as follows:

On [Baidu AI Studio Community - AI Learning and Training Platform](https://aistudio.baidu.com/paddlex/commercialization), under the "开源模型产线部署序列号咨询与获取" (open-source pipeline deployment serial number inquiry and acquisition) section, select "立即获取" (acquire now) as shown in the following image:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-1.png"/>

Select the pipeline you wish to deploy and click "获取" (acquire). Afterwards, you can find the acquired serial number in the "开源产线部署SDK序列号管理" (open-source pipeline deployment SDK serial number management) section at the bottom of the page:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipeline_deploy/image-2.png"/>

**Please note**: Each serial number can only be bound to a unique device fingerprint and can only be bound once. This means that if a user deploys a pipeline on different machines, a separate serial number must be prepared for each machine. **The high-stability serving solution is completely free.** PaddleX's authentication mechanism is targeted to count the number of deployments across various pipelines and provide pipeline efficiency analysis for our team through data modeling, so as to optimize resource allocation and improve the efficiency of key pipelines. It is important to note that the authentication process only uses non-sensitive information such as disk partition UUIDs, and PaddleX does not collect sensitive data such as device telemetry data. Therefore, in theory, **the authentication server cannot obtain any sensitive information**.

### 2.3 Adjust Configurations

The PaddleX high-stability serving solution is built on NVIDIA Triton Inference Server, allowing users to modify the configuration files of Triton Inference Server.

In the `server/model_repo/{endpoint name}` directory of the high-stability serving SDK, you can find one or more `config*.pbtxt` files. If a `config_{device type}.pbtxt` file exists in the directory, please modify the configuration file corresponding to the desired device type. Otherwise, please modify `config.pbtxt`.

A common requirement is to adjust the number of execution instances for horizontal scaling. To achieve this, you need to modify the `instance_group` setting in the configuration file, using `count` to specify the number of instances placed on each device, `kind` to specify the device type, and `gpus` to specify the GPU IDs. An example is as follows:

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

### 2.4 Run the Server

The machine used for deployment needs to have Docker Engine version 19.03 or higher installed.

First, pull the Docker image as needed:

- Image supporting deployment with NVIDIA GPU (the machine must have NVIDIA drivers that support CUDA 11.8 installed):

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.0.0rc0-gpu
    ```

- CPU-only Image:

    ```bash
    docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:paddlex3.0.0rc0-cpu
    ```

With the image prepared, navigate to the `server` directory and execute the following command to run the server:

```bash
docker run \
    -it \
    -e PADDLEX_HPS_DEVICE_TYPE={deployment device type} \
    -e PADDLEX_HPS_SERIAL_NUMBER={serial number} \
    -e PADDLEX_HPS_UPDATE_LICENSE=1 \
    -v "$(pwd)":/workspace \
    -v "${HOME}/.baidu/paddlex/licenses":/root/.baidu/paddlex/licenses \
    -v /dev/disk/by-uuid:/dev/disk/by-uuid \
    -w /workspace \
    --rm \
    --gpus all \
    --network host \
    --shm-size 8g \
    {image name} \
    /bin/bash server.sh
```

- The deployment device type can be `cpu` or `gpu`, and the CPU-only image supports only `cpu`.
- If CPU deployment is required, there is no need to specify `--gpus`.
- The above commands can only be executed properly after successful activation. PaddleX offers two activation methods: online activation and offline activation. They are detailed as follows:

    - Online activation: Set `PADDLEX_HPS_UPDATE_LICENSE` to `1` for the first execution to enable the program to automatically update the license and complete activation. When executing the command again, you may set `PADDLEX_HPS_UPDATE_LICENSE` to `0` to avoid online license updates.
    - Offline Activation: Follow the instructions in the serial number management section to obtain the machine’s device fingerprint. Bind the serial number with the device fingerprint to obtain the certificate and complete the activation. For this activation method, you need to manually place the certificate in the `${HOME}/.baidu/paddlex/licenses` directory on the machine (if the directory does not exist, you will need to create it). When using this method, set `PADDLEX_HPS_UPDATE_LICENSE` to `0` to avoid online license updates.

- It is necessary to ensure that the `/dev/disk/by-uuid` directory on the host machine exists and is not empty, and that this directory is correctly mounted in order to perform activation properly.
- If you need to enter the container for debugging, you can replace `/bin/bash server.sh` in the command with `/bin/bash`. Then execute `/bin/bash server.sh` inside the container.
- If you want the server to run in the background, you can replace `-it` in the command with `-d`. After the container starts, you can view the container logs with `docker logs -f {container ID}`.
- Add `-e PADDLEX_USE_HPIP=1` to use the PaddleX high-performance inference plugin to accelerate the pipeline inference process. However, please note that not all pipelines support using the high-performance inference plugin. Please refer to the [PaddleX High-Performance Inference Guide](./high_performance_inference.en.md) for more information.

You may observe output similar to the following:

```text
I1216 11:37:21.601943 35 grpc_server.cc:4117] Started GRPCInferenceService at 0.0.0.0:8001
I1216 11:37:21.602333 35 http_server.cc:2815] Started HTTPService at 0.0.0.0:8000
I1216 11:37:21.643494 35 http_server.cc:167] Started Metrics Service at 0.0.0.0:8002
```

### 2.5 Invoke the Service

Currently, only the Python client is supported for calling the service. Supported Python versions are 3.8 to 3.12.

Navigate to the `client` directory of the high-stability serving SDK, and run the following command to install dependencies:

```bash
# It is recommended to install in a virtual environment
python -m pip install -r requirements.txt
python -m pip install paddlex_hps_client-*.whl
```

The `client.py` script in the `client` directory contains examples of how to call the service and provides a command-line interface.

The services deployed using the high-stability serving solution offer the primary operations that match those of the basic serving solution. For each primary operation, the endpoint names and the request and response data fields are consistent with the basic serving solution. Please refer to the "Development Integration/Deployment" section in the tutorials for each pipeline. The tutorials for each pipeline can be found [here](../pipeline_usage/pipeline_develop_guide.en.md).
