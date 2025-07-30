---
comments: true
---

# PaddleX High Stability Serving

This project provides a high-stability serving solution, consisting of two main components: `server_env` and `sdk`.`server_env` is responsible for building multiple Docker images that include Triton Inference Server, providing the runtime environment for pipeline servers.`sdk` is used to package the pipeline SDK, including both server and client code for various model pipelines. As shown in the following figure:

<img src="https://github.com/cuicheng01/PaddleX_doc_images/blob/main/images/hps/hps_workflow_en.png?raw=true"/>


**Note: This project relies on the following environment configurations:**

- **Operating System**: Linux
- **Docker Version**: `>= 20.10.0` (Used for image building and deployment)
- **CPU Architecture**: x86-64

This  document  mainly introduces how to set up a high stability serving environment and package related materials using the scripts provided by this project. The overall process consists of two main stages:

1. Image Building: Build Docker images that include Triton Inference Server. In this stage, requirement versions are locked to ensure reproducibility and stability of the deployment images.

2. Pipeline Material Packaging: Package the client and server code for each model pipeline, making it easier for subsequent deployment and integration.

To learn how to start the server and invoke services using the built images and packaged SDK, please refer to the [PaddleX Serving Guide](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_deploy/serving.html) for detailed instructions.


## 1. Image Building

This stage mainly introduces the overall process and key steps of image building.

Image Building Steps:

1. Build a requirement collection image. (Optional)
2. Freeze requirement versions to improve the reproducibility of deployment image building. (Optional)
3. Build the deployment image based on the frozen requirement information to generate the final deployment image and provide image support for subsequent pipeline execution. 

**If you do not need to modify requirement-related information, you can skip to [1.3 Building Image](./README_en.md#13-building-image) to build the deployment image using cached requirement information.**

## 1.1 Build the Requirement Collection Image (Optional)

Navigate to the `server_env` directory and run follow script for building the requirement collection image in this directory. 

```bash
./scripts/prepare_rc_image.sh
```

This script builds a requirement collection image for each device type. The image includes Python 3.10 and [pip-tools](https://github.com/jazzband/pip-tools). [1.2 Freeze Requirement (Optional)](./README_en.md#12-freeze-requirement-optional) will be based on this image. After the build is complete, two images: `paddlex-hps-rc:gpu` and `paddlex-hps-rc:cpu` will be generated. If you encounter network issues, you can specify other pip sources through the `-p` parameter. If not specified, the default source https://pypi.org/simple will be used.If you encounter issues pulling the base image during the build process, please refer to the relevant solutions in the [FAQ](./README_en.md#3-faq).

## 1.2 Freeze Requirement (Optional)

To enhance the reproducibility of the build, this step freeze requirement to exact versions. Please switch to the `server_env` directory and run the following script:

```bash
./scripts/freeze_requirements.sh
```

This script uses `pip-tools compile` to parse the source requirement files and generate a series of `.txt` files (such as `requirements/gpu.txt`, `requirements/cpu.txt`, etc.). These files will serve as version requirement for [1.3 Building Image](./README_en.md#13-building-image).

## 1.3 Building Image

If you need to build the GPU image, make sure to place the following two installation packages in the `server_env` directory in advance:

<table>
  <thead>
    <tr>
      <th>Package Name</th>
      <th>Version</th>
      <th>Download URL</th>
      <th>File Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>cuDNN</strong></td>
      <td>v8.9.7-CUDA 11.x</td>
      <td><a href="https://developer.nvidia.cn/rdp/cudnn-archive">NVIDIA cuDNN Archive</a></td>
      <td>Local Installer for Linux x86_64 (Tar)</td>
    </tr>
    <tr>
      <td><strong>TensorRT</strong></td>
      <td>8.6-GA</td>
      <td><a href="https://developer.nvidia.com/nvidia-tensorrt-8x-download">TensorRT 8.x Download Page</a></td>
      <td>TensorRT 8.6 GA for Linux x86_64 and CUDA 11.x TAR Package</td>
    </tr>
  </tbody>
</table>

For Triton Inference Server, a precompiled version will be automatically downloaded during the build process, so manual download is not required. To build a GPU image, run the following command:

```bash
./scripts/build_deployment_image.sh -k gpu -t latest-gpu
```

Build image script supports the following configuration options:

<table>
<thead>
<tr>
<th>Name</th>
<th>Descrition</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>-k</code></td>
<td>Specifies the device type for the image. Supported values: <code>gpu</code> or <code>cpu</code>.</td>
</tr>
<tr>
<td><code>-t</code></td>
<td>Sets the image tag. Default: <code>latest:${DEVICE}</code>.</td>
</tr>
<tr>
<td><code>-p</code></td>
<td>Python package index URL. If not specified, defaults to <code>https://pypi.org/simple</code>.</td>
</tr>
</tbody>
</table>

If the basic image cannot be pulled, please refer to the solutions in the [FAQ](./README_en.md#3-faq).

After run successfully, the command line will display the following message:

```text
 => => exporting to image                                                         
 => => exporting layers                                                      
 => => writing image  sha256:ba3d0b2b079d63ee0239a99043fec7e25f17bf2a7772ec2fc80503c1582b3459   
 => => naming to ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-gpu   
```

To build both GPU and CPU images  run the following command:

```bash
./scripts/prepare_deployment_images.sh
```

## 2. Pipeline Material Packaging

This stage mainly introduces how to package pipeline materials. This function is provided in the `sdk` directory, which offers corresponding client and server code implementations for each pipeline:

- `client`: Responsible for invoking the model services.
- `server`: Deployed using the images built in [1. Image Building](./README_en.md#1-image-building), serving as the runtime environment for model services.

Before packaging the pipeline materials, you need to switch to the `sdk` directory and run the `scripts/assemble.sh` script in this directory for  packaging. For example, to package the general OCR pipeline, run:

```bash
./scripts/assemble.sh OCR
```

The parameters for the packaging script are described as follows:

<table>
<thead>
<tr>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline_names</code></td>
<td>Specifies the names of the pipelines to be packaged. Can be empty or include multiple names. For example, the general OCR pipeline is <code>OCR</code>.</td>
</tr>
<tr>
<td><code>--all</code></td>
<td>Packages all pipelines. Cannot be used together with <code>pipeline_names</code>.</td>
</tr>
<tr>
<td><code>--no-server</code></td>
<td>Excludes the server code from the package.</td>
</tr>
<tr>
<td><code>--no-client</code></td>
<td>Excludes the client code from the package.</td>
</tr>
</tbody>
</table>

After run successfully, the packaged  will be stored in the `/output` directory.

## 3. FAQ

**1. Failed to pull the base Docker image during build?**

This issue may occur due to network connectivity problems or restricted access to Docker Hub. You can add trusted domestic mirror registry URLs to your local Docker configuration file at `/etc/docker/daemon.json` to improve download speed and stability. If this does not resolve the issue, consider manually downloading the base image from the official source or other trusted third-party source.


**2. Timeout when installing Python requirement during image build?**

Network issues may cause slow download speeds or connection failures when pip retrieves packages from the official source.
When running the image build scripts, you can use the `-p` parameter to specify an alternative Python package index URL. For example, to use the Tsinghua mirror for the requirement collection image:

```bash
./scripts/prepare_rc_image.sh -p  https://pypi.tuna.tsinghua.edu.cn/simple
```