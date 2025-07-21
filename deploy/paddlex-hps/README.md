---
comments: true
---

# PaddleX 高稳定性服务化部署

本项目 提供一套高稳定性部署服务，它由 server_env 与 sdk 两个子文件夹组成，server_env 部分用于构建包含 Triton Server 的多个镜像，为后续模型产线server提供运行环境。sdk 部分用于打包产线 SDK，提供各模型产线的 server 和 client 代码 ，便于快速调用模型服务。

**请注意，当前该方案仅支持 Linux 系统。**

PaddleX 高稳定性服务化部署示意图

<img src="https://github.com/boomercat/doc_image/blob/main/hps_project.drawio.png?raw=true" />

## 1. 镜像构建

若您为第一次构建镜像，则需从此部分开始。若您曾按照此步骤构建过镜像，则从[2.2 产线调用](#22-产线调用)开始。

本阶段主要用于收集和固定本项目运行所需的环境信息与依赖配置，确保在不同设备类型（GPU / CPU）和运行环境下的一致性和稳定性。而后基于固定好的环境进行镜像构建，为后续产线调用提供部署服务。

镜像构建主要包括以下核心步骤，其中环境收集及固定是部署流程的前置阶段：
- 构建依赖收集镜像(RC Image)
- 冻结依赖
- 构建部署镜像

### 1.1构建依赖收集镜像
执行server_env文件夹下的依赖收集脚本，如需指定pip源需添加`-p`参数，如不指定默认为https://pypi.org/simple。
```bash
./scripts/prepare_rc_image.sh
``` 
该脚本会基于 Dockerfile 的 rc 阶段构建一个轻量镜像，该镜像包含 Python 3.10 及 pip-tools 工具。后续步骤将基于该镜像完成依赖冻结。构建完成后，分别生成 paddlex-hps-rc:gpu 和 paddlex-hps-rc:cpu 两个镜像。

### 1.2 冻结依赖
为了确保镜像环境稳定，必须将依赖锁定到精确版本。如需构建GPU镜像，需提前将<a href="https://developer.nvidia.cn/rdp/cudnn-archive">cuDNN8.9.7-CUDA11.x 安装包</a>和<a href="https://developer.nvidia.com/nvidia-tensorrt-8x-download">TensorRT 8.6.1.6-Ubuntu20.04 安装包</a>放在server_env目录下。而涉及到的 Triton Server 是提前根据项目特性编译好托管的zip包，直接在Dockerfile中通过wget下载，故不需要手动下载。在 1.1 阶段构建的两个镜像基础上进行依赖冻结操作。
```bash
./script/free_requirements.sh
```

执行 freeze_requirements.sh 脚本后，会启动前面构建的 RC 镜像，并在容器中运行 _freeze_requirements.sh。该脚本调用 pip-tools compile，解析响应的依赖源文件（通用的requirements/app.in、指定设备的 cpu.in/gpu.in 文件和高性能的*_hpi.in文件），最终生成两类依赖文件：
- 基础依赖：如 requirements/gpu.txt 或 requirements/cpu.txt。
- HPI 依赖：*_hpi.txt。

以上依赖文件为 1.3节 的镜像构建提供了版本约束。

### 1.3镜像构建

基于1.2的依赖冻结完成后，我们构建最终部署镜像：
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
<td>构建镜像的设备类型，可选为<code>gpu</code>或<code>cpu</code>。<br /><code>gpu</code> 镜像内包含TensorRT、cudnn、Triton Server等，<code>cpu</code> 镜像内包含Triton Server等。</td>
</tr>

<tr>
  <td><code>-t</code></td>
  <td>镜像标签，默认为 <code>latest:${DEVICE}</code> </td>
</tr>
<tr>
  <td><code>-p</code></td>
  <td>pip下载源，如不指定默认为<code>https://pypi.org/simple。</code></td>
</tr>
</tbody>
</table>

得到如下字样
```text
 => exporting to image                                                         
 => => exporting layers                                                      
 => => writing image  sha256:ba3d0b2b079d63ee0239a99043fec7e25f17bf2a7772ec2fc80503c1582b3459   
 => => naming to ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/hps:latest-gpu   
```
如需批量构建 GPU 和 CPU 镜像，可以如下指令使用此脚本得到两个镜像：
```bash
./srcipts/prepare_deployment_images.sh
```

## 2. 产线SDK打包及调用

SDK 模块为多个模型产线提供了统一的打包等功能。同时，该模块为每个产线提供对应的 client 和 server 代码结构：
- client 部分 ：用于调用模型服务，提供统一的 SDK 接口。
- server 部分 ：基于第一阶段构建的镜像作为运行环境，用于部署模型服务。

通过 SDK 模块，可以将不同产线的 client 与 server 代码打包为独立的 SDK 发布包，便于集成和部署。

### 2.1 sdk 打包

为了便于发布与部署，SDK 模块支持将不同产线的 client 和 server 代码打包为压缩包或whl。打包流程由 scripts/assemble.sh 脚本控制。

```bash
./scripts/assemble.sh  OCR
```
打包可选的参数包括
<table>
<thead>
<tr>
<th>名称</th>
<th>说明</th>
<th>类型</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>pipeline_names</code></td>
<td>构建镜像的设备类型，需要打包的具体产线名称，例如OCR产线为<code>OCR</code>，必选参数。</td>
<td>str</td>
</tr>
<tr>
  <td><code>--all</code></td>
  <td>打包全部产线，与<code>pipeline_name</code>不可共用。</td>
  <td>bool</td>
</tr>
<tr>
  <td><code>--no_server</code></td>
  <td>不打包产线中的<code>server</code>代码  </td>
  <td>bool</td>
</tr>
<tr>
  <td><code>--no_client</code></td>
  <td>不打包产线中的<code>client</code>代码</td>
  <td>bool</td>
</tr>
</tbody>
</table>

调用后默认存储到当前目录/output路径下。
### 2.2 产线调用

具体的产线调用详情可参考[PaddleX服务化部署](../../docs/pipeline_deploy/serving.md#23-运行服务器)文档。
