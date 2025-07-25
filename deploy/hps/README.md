---
comments: true
---

# PaddleX 高稳定性服务化部署

本项目提供一套高稳定性服务化部署方案，它由 `server_env` 与 `sdk` 两个目录组成。`server_env` 部分用于构建包含 Triton Inference Server 的多种镜像，为后续模型产线 server 提供运行环境；`sdk` 部分用于打包产线 SDK，提供各模型产线的 server 和 client 代码。如下图所示：

<img src="https://github.com/boomercat/PaddleX_doc_images/blob/main/images/hps/hps_workflow.png?raw=true" />

**请注意，本项目依赖于如下环境配置：**

- **操作系统**：Linux
- **Docker 版本**：`>= 20.10.0`，用于镜像构建和部署
- **CPU 架构**：x86-64 

本文档主要介绍如何基于本项目提供的脚本完成高稳定性服务化部署环境搭建与物料打包。整体流程分为两个阶段：

1. 镜像构建：构建包含 Triton Inference Server 的镜像。在这一阶段中，依赖版本被锁定以提升部署镜像构建的可重现性。
2. 产线物料打包：将各模型产线的客户端和服务端代码进行打包，便于后续部署与集成使用。

如需了解如何使用构建好的镜像与打包好的 SDK 启动服务器和调用服务，可参考 [PaddleX 服务化部署指南](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/serving.html)。

## 1. 镜像构建

本阶段主要介绍镜像构建的整体流程及关键步骤。

镜像构建步骤：

1. 构建依赖收集镜像。
2. 锁定依赖版本，提升部署镜像构建的可重现性。
3. 构建部署镜像，基于已锁定的依赖信息，构建最终的部署镜像，为后续的产线运行提供镜像支持。

### 1.1 构建依赖收集镜像

执行 `server_env` 目录下的依赖收集脚本。

```bash
./scripts/prepare_rc_image.sh
```

该脚本会为每种设备类型构建一个用于依赖收集的镜像，镜像包含 Python 3.10 以及 [pip-tools](https://github.com/jazzband/pip-tools) 工具。[1.2 锁定依赖](./README.md#12-锁定依赖) 将基于该镜像完成。构建完成后，将分别生成 `paddlex-hps-rc:gpu` 和 `paddlex-hps-rc:cpu` 两个镜像。如果遇到网络问题，可以通过 `-p` 参数指定其他 pip 源。如果不指定，默认使用 https://pypi.org/simple。

### 1.2 锁定依赖

为了使构建结果的可重现性更强，本步骤将依赖锁定到精确版本。执行如下脚本：

```bash
./script/freeze_requirements.sh
```

该脚本调用 `pip-tools compile` 解析依赖源文件，并最终生成一系列 `.txt` 文件（如 `requirements/gpu.txt`、`requirements/cpu.txt` 等），这些文件将为 [1.3 镜像构建](./README.md#13-镜像构建) 提供依赖版本约束。

### 1.3 镜像构建

在完成 1.2 锁定依赖后，如需构建 GPU 镜像，需提前将 [cuDNN 8.9.7-CUDA 11.x 安装包](https://developer.nvidia.cn/rdp/cudnn-archive) 和 [TensorRT 8.6.1.6-Ubuntu 20.04 安装包](https://developer.nvidia.com/nvidia-tensorrt-8x-download) 放在 `server_env` 目录下。对于 Triton Server，项目使用预先编译好的版本，将在构建镜像时自动下载，无需手动下载。以构建 GPU 镜像为例，执行以下命令：

```bash
./scripts/build_deployment_image.sh -k gpu -t latest-gpu 
```

构建镜像的参数配置项包括

<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>-k</code></td>
<td>指定镜像的设备类型，可选值为 <code>gpu</code> 或 <code>cpu</code>。</td>
</tr>
<tr>
<td><code>-t</code></td>
<td>镜像标签，默认为 <code>latest:${DEVICE}</code>。</td>
</tr>
<tr>
<td><code>-p</code></td>
<td>Python 包索引 URL，如不指定默认为 <code>https://pypi.org/simple</code>。</td>
</tr>
</tbody>
</table>

执行成功后，命令行会输出以下提示信息：

```text
 => => exporting to image                                                         
 => => exporting layers                                                      
 => => writing image  sha256:ba3d0b2b079d63ee0239a99043fec7e25f17bf2a7772ec2fc80503c1582b3459   
 => => naming to ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-gpu   
```

如需批量构建 GPU 和 CPU 镜像，可以执行以下命令：

```bash
./srcipts/prepare_deployment_images.sh
```

## 2. 产线物料打包

本阶段主要介绍 `sdk` 目录下为多个模型产线提供统一的打包功能。同时，该目录为每个产线提供对应的 client 和 server 代码实现：

- `client` 部分：用于调用模型服务。
- `server` 部分：以 [1. 镜像构建](#1-镜像构建) 阶段构建的镜像作为运行环境，用于部署模型服务。

打包可通过 `scripts/assemble.sh` 脚本执行，以打包通用 OCR 产线为例：

```bash
./scripts/assemble.sh OCR
```

打包脚本的参数说明如下：

<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline_names</code></td>
<td>需要打包的产线名称，可以为空或一次指定多个，例如通用 OCR 产线为 <code>OCR</code>。</td>
</tr>
<tr>
<td><code>--all</code></td>
<td>打包全部产线，与 <code>pipeline_names</code> 不可共用。</td>
</tr>
<tr>
<td><code>--no-server</code></td>
<td>不打包产线中的 server 代码。</td>
</tr>
<tr>
<td><code>--no-client</code></td>
<td>不打包产线中的 client 代码。</td>
</tr>
</tbody>
</table>

调用后存储到当前目录 `/output` 路径下。



## 3.FAQ

**1. 构建镜像时无法拉取 Docker 基础镜像**

由于网络连接问题或镜像源访问限制，可能会导致从 Docker Hub 拉取基础镜像失败。可尝试在本地 Docker 配置文件 `/etc/docker/daemon.json` 中添加国内可信镜像仓库地址，以提升镜像下载速度和稳定性。如果上述方法仍无法解决，可尝试从官方或可信第三方渠道手动下载镜像文件。


**2. 镜像构建过程中出现安装 Python 依赖时超时？**

可能由于网络问题，pip 从官方源下载依赖速度过慢或连接失败。在执行构建镜像脚本时，使用 `-p` 参数指定国内 Python 包索引 URL，以构建依赖收集镜像脚本使用清华镜像源为例：

```bash
./scripts/prepare_rc_image.sh -p https://pypi.tuna.tsinghua.edu.cn/simple
```
