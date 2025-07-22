---
comments: true
---

# PaddleX 高稳定性服务化部署

本项目提供一套高稳定性部署服务，它由 `server_env` 与 `sdk` 两个目录组成，`server_env` 部分用于构建包含 Triton Server 的多种镜像，为后续模型产线 server 提供运行环境。`sdk` 部分用于打包产线 SDK，提供各模型产线的 server 和 client 代码 ，便于快速调用模型服务。如下图所示：
PaddleX 高稳定性服务化部署示意图

<img src="https://github.com/boomercat/doc_image/blob/main/hps_project.drawio.png?raw=true" />

**请注意，在执行本项目前，请确保具备以下基础环境：**
- **操作系统**：Linux（推荐 Ubuntu 20.04+）
- **Docker**：`>= 20.10.0`，用于镜像构建和部署
- **NVIDIA Container Toolkit**（如需 GPU 镜像）
  > 安装指南：[NVIDIA 官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **CPU 架构**：x86_64 
## 1. 镜像构建

本阶段主要介绍镜像构建的整体流程及其关键步骤，目的是在明确依赖和运行环境的基础上，构建可用于部署的镜像，保证运行稳定性。

镜像构建步骤：
1. 依赖收集镜像：收集本项目所需的依赖，确保环境配置完整。

2. 冻结依赖：将上一步骤中依赖收集的版本固定，生成依赖清单文件，避免后续构建过程中出现版本不一致导致的问题。

3. 构建部署镜像：基于已冻结的依赖，构建最终的部署镜像，为后续的产线运行提供镜像支持。

### 1.1 构建依赖收集镜像
执行 server_env 文件夹下的依赖收集脚本，如果遇到网络问题，可以通过 `-p` 参数指定其他 pip 源。如果不指定，默认为 https://pypi.org/simple。
```bash
./scripts/prepare_rc_image.sh
``` 
该脚本会构建一个用于依赖收集的镜像，包含 Python 3.10 以及 <a href="https://github.com/jazzband/pip-tools">pip-tools</a> 工具。1.2 冻结依赖步骤将基于该镜像完成。构建完成后，将分别生成 paddlex-hps-rc:gpu 和 paddlex-hps-rc:cpu 两个镜像。

### 1.2 冻结依赖

为了确保镜像环境稳定，必须将依赖锁定到精确版本。如需构建GPU镜像，需提前将<a href="https://developer.nvidia.cn/rdp/cudnn-archive">cuDNN8.9.7-CUDA11.x 安装包</a>和<a href="https://developer.nvidia.com/nvidia-tensorrt-8x-download">TensorRT 8.6.1.6-Ubuntu20.04 安装包</a>放在 `server_env` 目录下。对于 Triton Server，项目使用预先编译好的版本，将在构建镜像时自动下载，无需手动下载。依赖冻结基于 1.1 阶段构建的两个镜像进行。
```bash
./script/free_requirements.sh
```
执行 `freeze_requirements.sh` 脚本后，会启动前面构建的依赖收集镜像，并在容器中运行 `_freeze_requirements.sh`。该脚本调用 `pip-tools compile`，解析依赖源文件，并最终生成一系列 .txt 文件（如 `requirements/gpu.txt`、`requirements/cpu.txt` 等）。这些文件将为 1.3 节的镜像构建提供版本约束。

### 1.3 镜像构建

在完成 1.2 依赖冻结 后，以构建 GPU 镜像为例，执行以下命令：

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
<td>指定镜像的设备类型，可选值为 <code>gpu</code> 或 <code>cpu</code></td>
</tr>
<tr>
<td><code>-t</code></td>
<td>镜像标签，默认为 <code>latest:${DEVICE}</code> </td>
</tr>
<tr>
<td><code>-p</code></td>
<td>Python 包索引 URL，如不指定默认为 <code>https://pypi.org/simple</code></td>
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

## 2. 产线 SDK 打包及调用

本阶段为多个模型产线提供统一的打包功能。同时，该模块为每个产线提供对应的 `client` 和 `server` 代码实现：

- `client` 部分：用于调用模型服务，提供统一的 SDK 接口。
- `server` 部分：基于 [1. 镜像构建](#1-镜像构建) 阶段构建的镜像作为运行环境，用于部署模型服务。

### 2.1 SDK 打包

为了便于发布与部署，SDK 模块支持将不同产线的 `client` 和 `server` 代码打包。打包可通过 `scripts/assemble.sh` 脚本执行,以打包 OCR 产线为例：

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
<td>需要打包的产线名称，可以为空或一次指定多个，例如 OCR 产线为<code>OCR</code>。</td>
</tr>
<tr>
<td><code>--all</code></td>
<td>打包全部产线，与<code>pipeline_names</code>不可共用。</td>
</tr>
<tr>
<td><code>--no-server</code></td>
<td>不打包产线中的<code>server</code>代码。</td>
</tr>
<tr>
<td><code>--no-client</code></td>
<td>不打包产线中的<code>client</code>代码。</td>
</tr>
</tbody>
</table>

调用后存储到当前目录/output路径下。

### 2.2 产线调用

具体的产线调用详情可参考[PaddleX 服务化部署指南](../../docs/pipeline_deploy/serving.md#23-运行服务器)了解如何启动服务器与调用产线服务。


## 3.FAQ

#### 1. 构建镜像时无法拉取 Docker 基础镜像？

可能由于网络问题或镜像源限制，导致从 Docker Hub 拉取基础镜像失败。可尝试更换镜像加速器，在本地 Docker 配置文件 `/etc/docker/daemon.json` 中添加国内镜像源，或尝试直接手动拉取镜像。


#### 2. 镜像构建过程中出现安装 Python 依赖时超时？

可能由于网络问题，pip 从官方源下载依赖速度过慢或连接失败。在执行依赖收集或构建镜像时，使用 -p 参数指定国内 Python 包索引 URL，例如清华源镜像：

```bash
./scripts/prepare_rc_image.sh -p https://mirrors.aliyun.com/pypi/simple/
```

#### 3.镜像构建过程中 Triton Server 下载失败怎么办？

项目使用预先编译好的 Triton Server 包，默认通过 wget 从托管地址下载，若网络不通则失败。尝试通过手动wget下载Dockerfile中的 Triton Server 压缩包 ，并放置在 `server_env` 目录下，需要将 Dockerfile 内容修改，将原来的 wget 操作 `RUN wget` ... 替换为：

```bash
COPY tritonserver-2.15.0-${DEVICE_TYPE}.zip /paddlex/
RUN unzip "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}.zip" -d /paddlex \
    && mv "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}" /paddlex/tritonserver \
    && rm "/paddlex/tritonserver-2.15.0-${DEVICE_TYPE}.zip"
```

完成后重新执行构建脚本。


