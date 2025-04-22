---
comments: true
---

# PaddleX本地安装教程
> ❗安装 PaddleX 前请先确保您有基础的 <b>Python 运行环境</b>（注：当前支持Python 3.8 ～ Python 3.12下运行）。

## 1. 快速安装
欢迎您使用飞桨低代码开发工具PaddleX，在我们正式开始本地安装之前，请首先明确您的开发需求，并根据您的需求选择合适的安装模式。
PaddleX为您提供了两种安装模式：<b>Wheel包安装</b>和<b>插件安装</b>，下面分别对其应用场景进行介绍：

### 1.1 Wheel包安装模式
若您使用PaddleX的应用场景为<b>模型推理与集成</b> ，那么推荐您使用<b>更便捷</b>、<b>更轻量</b>的Wheel包安装模式。

快速安装轻量级的Wheel包之后，您即可基于PaddleX支持的所有模型进行推理，并能直接集成进您的项目中。

参考[飞桨PaddlePaddle本地安装教程](paddlepaddle_install.md)安装飞桨后，您可直接执行如下指令快速安装PaddleX的Wheel包：

> ❗ 注：请务必保证 PaddlePaddle 安装成功，安装成功后，方可进行下一步。

```bash
pip install paddlex==3.0.0rc1
```

### 1.2 插件安装模式
若您使用PaddleX的应用场景为<b>二次开发</b> （例如重新训练模型、微调模型、自定义模型结构、自定义推理代码等），那么推荐您使用<b>功能更加强大</b>的插件安装模式。

安装您需要的PaddleX插件之后，您不仅同样能够对插件支持的模型进行推理与集成，还可以对其进行模型训练等二次开发更高级的操作。

PaddleX支持的模型训练相关插件如下，请您根据开发需求，确定所需的一个或多个插件名称：

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
<td>文档图像方向分类<br>文本图像矫正<br>文本检测<br>文本行方向分类<br>文本识别</td>
<td><code>PaddleOCR</code><br><code>PaddleClas</code></td>
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

当您使用官方 Docker 镜像安装时，其中<b>已经内置了 PaddlePaddle、PaddleX（包括wheel包和所有插件）</b>，并配置好了相应的CUDA环境，<b>您获取 Docker 镜像并启动容器即可开始使用</b>。<b>请注意，PaddleX 官方 Docker 镜像与飞桨框架官方 Docker 镜像不同，后者并没有预装 PaddleX。</b>

当您使用自定义方式安装时，需要先安装飞桨 PaddlePaddle 框架，随后获取 PaddleX 源码，最后选择PaddleX的安装模式。

> ❗ 无需关注物理机上的 CUDA 版本，只需关注显卡驱动程序版本。

### 2.1 基于Docker获取PaddleX
参考下述命令，使用 PaddleX 官方 Docker 镜像，创建一个名为 `paddlex` 的容器，并将当前工作目录映射到容器内的 `/paddle` 目录。

若您使用的 Docker 版本 >= 19.03，请执行：

```bash
# 对于 CPU 用户
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc1-paddlepaddle3.0.0-cpu /bin/bash

# 对于 GPU 用户
# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
docker run --gpus all --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc1-paddlepaddle3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6 /bin/bash
```

* 若您使用的 Docker 版本 <= 19.03 但 >= 17.06，请执行：

<details><summary> 点击展开</summary>

<pre><code class="language-bash"># 对于 CPU 用户
docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc1-paddlepaddle3.0.0-cpu /bin/bash

# 对于 GPU 用户
# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
nvidia-docker run --name paddlex -v $PWD:/paddle --shm-size=8g --network=host -it ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlex/paddlex:paddlex3.0.0rc1-paddlepaddle3.0.0-gpu-cuda11.8-cudnn8.6-trt8.5 /bin/bash

</code></pre></details>

* 若您使用的 Docker 版本 <= 17.06，请升级 Docker 版本。

* 若您想更深入了解 Docker 的原理或使用方式，请参考 [Docker官方网站](https://www.docker.com/) 或 [Docker官方教程](https://docs.docker.com/get-started/)。CUDA12以上版本正在支持中，敬请期待。

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
pip install -e ".[base]"
```
<b>若您选择插件安装模式</b>，并且您需要的插件名称为 PaddleXXX（可以有多个），请执行以下命令：

```bash
cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e ".[base]"

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

### 2.3 选择性安装依赖

PaddleX 的功能丰富，而不同的功能需要的依赖也不尽相同。将 PaddleX 中不需要安装插件即可使用的功能归类为“基础功能”。PaddleX 官方 Docker 镜像预置了基础功能所需的全部依赖；使用上文介绍的 `pip install "...[base]"` 的安装方式也将安装基础功能需要的所有依赖。如果您只专注于 PaddleX 的某一项功能，且希望保持安装的依赖的体积尽可能小，可以通过指定“依赖组”的方式，选择性地安装依赖：

```bash
# 以仅安装 OCR 类基础功能为例
# 安装预编译的 wheel 包
pip install "/url/of/wheel[ocr]"
# 从源码安装
pip install -e ".[ocr]"

# 也可以同时指定多个依赖组
pip install -e ".[ocr,cv]"
```

PaddleX 目前提供如下依赖组：

| 依赖组名称 | 对应的功能 |
| - | - |
| `base` | PaddleX 的所有基础功能。 |
| `cv` | CV 产线的基础功能。 |
| `multimodal` | 多模态产线的基础功能。 |
| `ie` | 信息抽取产线的基础功能。 |
| `ocr` | OCR 类产线的基础功能。 |
| `speech` | 语音产线的基础功能。 |
| `ts` | 时序产线的基础功能。 |
| `video` | 视频产线的基础功能。 |
| `serving` | 服务化部署功能。安装此依赖组等效于安装 PaddleX 服务化部署插件；也可以通过 PaddleX CLI 安装服务化部署插件。 |
| `plugins` | 所有支持通过指定依赖组安装的插件提供的功能。 |
| `all` | PaddleX 的所有基础功能，以及所有支持通过指定依赖组安装的插件提供的功能。 |


每一条产线属于且仅属于一个依赖组；在各产线的使用文档中可以了解产线属于哪一依赖组。对于单功能模块，安装任意包含该模块的产线对应的依赖组后即可使用相关的基础功能。
