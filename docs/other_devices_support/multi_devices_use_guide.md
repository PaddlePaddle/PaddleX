简体中文 | [English](multi_devices_use_guide_en.md)

# PaddleX多硬件使用指南

本文档主要针对昇腾 NPU、昆仑 XPU、寒武纪 MLU、海光DCU 硬件平台，介绍 PaddleX 使用指南。

## 1、安装
### 1.1 PaddlePaddle安装
首先请您根据所属硬件平台，完成飞桨 PaddlePaddle 的安装，各硬件的飞桨安装教程如下：

昇腾 NPU：[昇腾 NPU 飞桨安装教程](./paddlepaddle_install_NPU.md)

昆仑 XPU：[昆仑 XPU 飞桨安装教程](./paddlepaddle_install_XPU.md)

寒武纪 MLU：[寒武纪 MLU 飞桨安装教程](./paddlepaddle_install_MLU.md)

海光 DCU：[海光 DCU 飞桨安装教程](./paddlepaddle_install_DCU.md)

### 1.2 PaddleX安装
欢迎您使用飞桨低代码开发工具PaddleX，在我们正式开始本地安装之前，请先明确您的开发需求，并根据您的需求选择合适的安装模式。

PaddleX为您提供了两种安装模式：Wheel包安装和插件安装，下面详细介绍这两种安装模式的应用场景和安装方法。

#### 1.2.1 Wheel包安装模式
若您使用PaddleX的应用场景为**模型推理与集成** ，那么推荐您使用**更便捷**、**更轻量**的Wheel包安装模式。

快速安装轻量级的Wheel包之后，您即可基于PaddleX支持的所有模型进行推理，并能直接集成进您的项目中。

安装飞桨后，您可直接执行如下指令快速安装PaddleX的Wheel包：

```
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```
#### 1.2.2 插件安装模式
若您使用PaddleX的应用场景为**二次开发** ，那么推荐您使用**功能更加强大**的插件安装模式。

安装您需要的PaddleX插件之后，您不仅同样能够对插件支持的模型进行推理与集成，还可以对其进行模型训练等二次开发更高级的操作。

PaddleX支持的插件如下，请您根据开发需求，确定所需的一个或多个插件名称：


<details>
  <summary>👉 <b>插件和产线对应关系（点击展开）</b></summary>

|模型产线|模块|对应插件|
|-|-|-|
|通用图像分类|图像分类|PaddleClas|
|通用目标检测|目标检测|PaddleDetection|
|通用语义分割|语义分割|PaddleSeg|
|通用实例分割|实例分割|PaddleDetection|
|通用OCR|文本检测<br>文本识别|PaddleOCR|
|通用表格识别|版面区域检测<br>表格结构识别<br>文本检测<br>文本识别|PaddleOCR<br>PaddleDetection|
|文档场景信息抽取v3|表格结构识别<br>版面区域检测<br>文本检测<br>文本识别<br>印章文本检测<br>文本图像矫正<br>文档图像方向分类|PaddleOCR<br>PaddleDetection<br>PaddleClas |
|时序预测|时序预测模块|PaddleTS|
|时序异常检测|时序异常检测模块|PaddleTS|
|时序分类|时序分类模块|PaddleTS|
|通用多标签分类|图像多标签分类|PaddleClas|
|小目标检测|小目标检测|PaddleDetection|
|图像异常检测|无监督异常检测|PaddleSeg|

</details>


若您需要安装的插件为PaddleXXX（可以有多个），在安装飞桨后，您可以直接执行如下指令快速安装PaddleX的对应插件：

```
# 下载 PaddleX 源码
git clone https://github.com/PaddlePaddle/PaddleX.git
cd PaddleX

# 安装 PaddleX whl
# -e：以可编辑模式安装，当前项目的代码更改，都会直接作用到已经安装的 PaddleX Wheel
pip install -e .

# 安装 PaddleX 插件
paddlex --install PaddleXXX
```
例如，您需要安装PaddleOCR、PaddleClas插件，则需要执行如下命令安装插件：

```
# 安装 PaddleOCR、PaddleClas 插件
paddlex --install PaddleOCR PaddleClas
```
若您需要安装全部插件，则无需填写具体插件名称，只需执行如下命令：

```
# 安装 PaddleX 全部插件
paddlex --install
```
插件的默认克隆源为  github.com，同时也支持 gitee.com 克隆源，您可以通过`--platform` 指定克隆源。

例如，您需要使用 gitee.com 克隆源安装全部PaddleX插件，只需执行如下命令：

```
# 安装 PaddleX 插件
paddlex --install --platform gitee.com
```
安装完成后，将会有如下提示：

```
All packages are installed.
```
## 2、使用
基于昇腾 NPU、寒武纪 MLU、昆仑 XPU、海光DCU 硬件平台的 PaddleX 模型产线开发工具使用方法与 GPU 相同，只需根据所属硬件平台，修改配置设备的参数，详细的使用教程可以查阅[PaddleX产线开发工具本地使用教程](../pipeline_usage/pipeline_develop_guide.md)
