---
comments: true
---

# PaddleX本地安装教程
> ❗安装 PaddleX 前请先确保您有基础的 <b>Python 运行环境</b>（注：当前支持Python 3.8 ～ Python 3.10下运行，更多Python版本适配中）。

## 1. 快速安装
欢迎您使用飞桨低代码开发工具PaddleX，在我们正式开始本地安装之前，请首先明确您的开发需求，并根据您的需求选择合适的安装模式。
PaddleX为您提供了两种安装模式：<b>Wheel包安装</b>和<b>插件安装</b>，下面分别对其应用场景进行介绍：

### 1.1 Wheel包安装模式
若您使用PaddleX的应用场景为<b>模型推理与集成</b> ，那么推荐您使用<b>更便捷</b>、<b>更轻量</b>的Wheel包安装模式。

快速安装轻量级的Wheel包之后，您即可基于PaddleX支持的所有模型进行推理，并能直接集成进您的项目中。

参考[飞桨PaddlePaddle本地安装教程](paddlepaddle_install.md)安装飞桨后，您可直接执行如下指令快速安装PaddleX的Wheel包：

> ❗ 注：请务必保证 PaddlePaddle 安装成功，安装成功后，方可进行下一步。

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl
```
### 1.2 插件安装模式
若您使用PaddleX的应用场景为<b>二次开发</b> （例如重新训练模型、微调模型、自定义模型结构、自定义推理代码等），那么推荐您使用<b>功能更加强大</b>的插件安装模式。

安装您需要的PaddleX插件之后，您不仅同样能够对插件支持的模型进行推理与集成，还可以对其进行模型训练等二次开发更高级的操作。

PaddleX支持的插件如下，请您根据开发需求，确定所需的一个或多个插件名称：

<details><summary>👉 <b>插件和产线对应关系（点击展开）</b></summary>

<table>
<thead>
<tr>
<th>模型产线</th>
<th>模块</th>
<th>对应插件</th>
</tr>
</thead>
<tbody>
<tr>
<td>通用图像分类</td>
<td>图像分类</td>
<td><code>PaddleClas</code></td>
</tr>
<tr>
<td>通用目标检测</td>
<td>目标检测</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>通用语义分割</td>
<td>语义分割</td>
<td><code>PaddleSeg</code></td>
</tr>
<tr>
<td>通用实例分割</td>
<td>实例分割</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>通用OCR</td>
<td>文本检测<br>文本识别</td>
<td><code>PaddleOCR</code></td>
</tr>
<tr>
<td>通用表格识别</td>
<td>版面区域检测<br>表格结构识别<br>文本检测<br>文本识别</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code></td>
</tr>
<tr>
<td>文档场景信息抽取v3</td>
<td>表格结构识别<br>版面区域检测<br>文本检测<br>文本识别<br>印章文本检测<br>文本图像矫正<br>文档图像方向分类</td>
<td><code>PaddleOCR</code><br><code>PaddleDetection</code><br><code>PaddleClas</code></td>
</tr>
<tr>
<td>时序预测</td>
<td>时序预测模块</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>时序异常检测</td>
<td>时序异常检测模块</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>时序分类</td>
<td>时序分类模块</td>
<td><code>PaddleTS</code></td>
</tr>
<tr>
<td>通用多标签分类</td>
<td>图像多标签分类</td>
<td><code>PaddleClas</code></td>
</tr>
<tr>
<td>小目标检测</td>
<td>小目标检测</td>
<td><code>PaddleDetection</code></td>
</tr>
<tr>
<td>图像异常检测</td>
<td>无监督异常检测</td>
<td><code>PaddleSeg</code></td>
</tr>
</tbody>
</table></details>



若您需要安装的插件为`PaddleXXX`，在参考[飞桨PaddlePaddle本地安装教程](paddlepaddle_install.md)安装飞桨后，您可以直接执行如下指令快速安装PaddleX的对应插件：

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX
pip install -e .
paddlex --install PaddleXXX  # 例如PaddleOCR
```

> ❗ 注：采用这种安装方式后，是可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel 包。

如果上述安装方式可以安装成功，则可以跳过接下来的步骤。

若您使用Linux操作系统，请参考[2. Linux安装PaddleX详细教程](#2-linux安装paddex详细教程)。其他操作系统的安装方式，敬请期待。

## 2. Linux安装PaddeX详细教程
使用Linux安装PaddleX时，我们<b>强烈推荐使用PaddleX官方Docker镜像安装</b>，当然也可使用其他自定义方式安装。

当您使用官方 Docker 镜像安装时，其中<b>已经内置了 PaddlePaddle、PaddleX（包括wheel包和所有插件）</b>，并配置好了相应的CUDA环境，<b>您获取 Docker 镜像并启动容器即可开始使用</b>。

当您使用自定义方式安装时，需要先安装飞桨 PaddlePaddle 框架，随后获取 PaddleX 源码，最后选择PaddleX的安装模式。

> ❗ 注：目前 PaddleX 仅支持 11.8 和 12.3 版本的 CUDA，请确保已安装的 Nvidia 驱动支持的上述 CUDA 版本。

### 2.1 基于Docker获取PaddleX
参考下述命令，使用 PaddleX 官方 Docker 镜像，创建一个名为 `paddlex` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

若您使用的 Docker 版本 >= 19.03，请执行：

```bash
# 对于 CPU 用户
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-cpu /bin/bash

# 对于 GPU 用户
# 对于 CUDA11.8 用户
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# 对于 CUDA12.3 用户
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
```

* 若您使用的 Docker 版本 <= 19.03 但 >= 17.06，请执行：

<details><summary> 点击展开</summary>

<pre><code class="language-bash"># 对于 CPU 用户
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-cpu /bin/bash

# 对于 GPU 用户
# 对于 CUDA11.8 用户
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

# 对于 CUDA12.3 用户
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc0-paddlepaddle3.0.0rc0-gpu-cuda12.3-cudnn9.0-trt8.6 /bin/bash
</code></pre></details>

* 若您使用的 Docker 版本 <= 17.06，请升级 Docker 版本。

* 若您想更深入了解 Docker 的原理或使用方式，请参考 [Docker官方网站](https://www.docker.com/) 或 [Docker官方教程](https://docs.docker.com/get-started/)。

### 2.2 自定义方式安装PaddleX
在安装之前，请确保您已经参考[飞桨PaddlePaddle本地安装教程](paddlepaddle_install.md)完成飞桨的本地安装。

#### 2.2.1 获取 PaddleX 源码
接下来，请使用以下命令从 GitHub 获取 PaddleX 最新源码：

```bash
git clone https://github.com/PaddlePaddle/PaddleX.git
```
如果访问 GitHub 网速较慢，可以从 Gitee 下载，命令如下：

```bash
git clone https://gitee.com/paddlepaddle/PaddleX.git
```
#### 2.2.2 安装PaddleX
获取 PaddleX 最新源码之后，您可以选择Wheel包安装模式或插件安装模式。

<b>若您选择Wheel包安装模式</b>，请执行以下命令：

```bash
cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .
```
<b>若您选择插件安装模式</b>，并且您需要的插件名称为 PaddleXXX（可以有多个），请执行以下命令：

```bash
cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .

# 安装 PaddleX 插件
paddlex --install PaddleXXX
```
例如，您需要安装PaddleOCR、PaddleClas插件，则需要执行如下命令安装插件：

```bash
# 安装 PaddleOCR、PaddleClas 插件
paddlex --install PaddleOCR PaddleClas
```
若您需要安装全部插件，则无需填写具体插件名称，只需执行如下命令：

```bash
# 安装 PaddleX 全部插件
paddlex --install
```
插件的默认克隆源为  github.com，同时也支持 gitee.com 克隆源，您可以通过`--platform` 指定克隆源。

例如，您需要使用 gitee.com 克隆源安装全部PaddleX插件，只需执行如下命令：

```bash
# 安装 PaddleX 插件
paddlex --install --platform gitee.com
```
安装完成后，将会有如下提示：

```
All packages are installed.
```
更多硬件环境的PaddleX安装请参考[PaddleX多硬件使用指南](../other_devices_support/multi_devices_use_guide.md)
