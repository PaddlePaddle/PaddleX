# PaddleX GUI开发模式快速上手
感谢使用PaddleX可视化客户端，通过本客户端，您可以实现图像分类、目标检测、实例分割和语义分割四大视觉任务模型的训练，裁剪及量化，以及模型在移动端/服务端的发布。

## 目录
- [快速安装](#快速安装)
  - [下载安装](#下载安装)
  - [安装推荐环境](#安装推荐环境)
- [视频教程](#视频教程)
- [文档教程](#文档教程)
  - [启动客户端](#1启动客户端)
  - [准备和导入数据](#2准备和导入数据)
  - [创建项目和任务](#3创建项目和任务)
  - [任务模型训练](#4任务模型训练)
  - [任务模型裁剪训练](#5任务模型裁剪训练)
  - [模型效果评估](#6模型效果评估)
  - [模型发布](#7模型发布)

## 快速安装
### 下载安装
下载地址：https://www.paddlepaddle.org.cn/paddlex
目前最新版本的GUI(Version 2.0.0)仅提供WIN和Linux版，暂未提供Mac版，若需在Mac上使用GUI，推荐安装Mac版历史版本Version 1.1.7
- WIN版下载后双击选择安装路径即可
- Mac/Linux版下载后解压即可

***注：安装/解压路径请务必在不包含中文和空格的路径下，否则可能会导致无法正确训练模型***

### 安装推荐环境

- **操作系统**：
  * Windows 10
  * Mac OS 10.13+
  * Ubuntu 18.04(Ubuntu暂只支持18.04)

***注：处理器需为x86_64架构，支持MKL。***

- **训练硬件**：  
  * **GPU**（仅Windows及Linux系统）：  
    推荐使用支持CUDA的NVIDIA显卡，例如：GTX 1070+以上性能的显卡
    Windows系统X86_64驱动版本>=411.31 
    Linux系统X86_64驱动版本>=410.48
    显存8G以上
  * **CPU**：PaddleX当前支持您用本地CPU进行训练，但推荐使用GPU以获得更好的开发体验。
  * **内存**：建议8G以上  
  * **硬盘空间**：建议SSD剩余空间1T以上（非必须）  

***注：PaddleX在Mac OS系统只支持CPU训练。Windows系统只支持单GPU卡训练。***

## 视频教程
用户可观看[图像分类](https://www.bilibili.com/video/BV1nK411F7J9?from=search&seid=3068181839691103009)、[目标检测](https://www.bilibili.com/video/BV1HB4y1A73b?from=search&seid=3068181839691103009)、[语义分割](https://www.bilibili.com/video/BV1qQ4y1Z7co?from=search&seid=3068181839691103009)、[实例分割](https://www.bilibili.com/video/BV1M44y1r7s6?from=search&seid=3068181839691103009)视频教程，并通过PaddleX可视化客户端完成四类任务。
<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133388877-b993a5a1-65ce-46a9-ada5-68d4e80fda3b.png" width="800" />
</p>

## 文档教程
### 1.启动客户端
如果系统是Mac OS 10.15.5及以上，在双击客户端icon后，需要在Terminal中执行 ```sudo xattr -r -d com.apple.quarantine /Users/username/PaddleX``` ，并稍等几秒来启动客户端，其中 /Users/username/PaddleX 为您保存PaddleX的文件夹路径*

### 2.准备和导入数据
- 准备数据
  - 在开始模型训练前，用户需要根据不同的任务类型，将数据标注为相应的格式。目前PaddleX支持【图像分类】、【目标检测】、【语义分割】、【实例分割】四种任务类型。  
  - 开发者可以参考PaddleX使用文档中的[数据标注](./docs/data/annotation)来进行数据标注和转换工作。如若开发者自行准备数据，请注意数据格式与PaddleX支持四种数据格式是否一致。

- 导入数据集

  ①数据标注完成后，需要根据不同的任务，将数据和标注文件，按照客户端提示更名并保存到正确的文件中。
  
  ②在客户端新建数据集，选择与数据集匹配的任务类型，并选择数据集对应的路径，将数据集导入。
  
  <p align="center">
    <img src="https://user-images.githubusercontent.com/53808988/133880285-2e29646a-89e0-4f97-a675-4586d7469216.jpg" width="800" />
  </p>
  
  ③选定导入数据集后，客户端会自动校验数据及标注文件是否合规，校验成功后，您可根据实际需求，将数据集按比例划分为训练集、验证集、测试集。
  
  ④您可在「数据分析」模块按规则预览您标注的数据集，双击单张图片可放大查看。
  
  <p align="center">
    <img src="https://user-images.githubusercontent.com/53808988/133880292-93d2f76b-1402-44bb-b84b-3a9ebc7c67c6.jpg" width="800" />
  </p>

### 3.创建项目和任务

- 创建项目

  ①在完成数据导入后，您可以点击「新建项目」创建一个项目。
  
  ②您可根据实际任务需求选择项目的任务类型，需要注意项目所采用的数据集也带有任务类型属性，两者需要进行匹配。
  <p align="center">
    <img src="https://user-images.githubusercontent.com/53808988/133880340-1da23b7c-249d-4175-b98e-62fbff9a1f7b.jpg" width="800" />
  </p>
- 项目开发

  ①数据选择：项目创建完成后，您需要选择已载入客户端并校验后的数据集，并点击下一步，进入参数配置页面。
  <p align="center">
    <img src="https://user-images.githubusercontent.com/53808988/133880374-157bc44a-6f64-45c5-bb3f-3608b8e85026.jpg" width="800" />
  </p>
  
  ②参数配置：主要分为**模型参数**、**训练参数**、**优化策略**三部分。您可根据实际需求选择模型结构、骨架网络及对应的训练参数、优化策略，使得任务效果最佳。
<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880390-3b97b772-2f7d-47bc-af9f-5943bca45177.jpg" width="800" />
</p>

### 4.任务模型训练

参数配置完成后，点击启动训练，模型开始训练并进行效果评估。

- 训练可视化：在训练过程中，您可通过VisualDL查看模型训练过程参数变化、日志详情，及当前最优的训练集和验证集训练指标。模型在训练过程中通过点击"中止训练"随时中止训练过程。
<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880453-e3fb6399-8545-44d7-9086-61889aa07d89.jpg" width="800" />
</p>

- 模型训练结束后，可选择进入『模型剪裁分析』或者直接进入『模型评估』。
<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880456-1c8bfa3b-757f-4927-b107-d3bb3bfcf529.jpg" width="800" />
</p>

> 模型训练是最容易出错的步骤，经常遇到的原因为 电脑无法联网下载预训练模型、显存不够。训练检测模型\实例分割模型对于显存要求较高，**建议用户通过在Windows/Mac/Ubuntu的命令行终端（Windows的cmd命令终端）执行`nvidia-smi`命令**查看显存情况，请不要使用系统自带的任务管理器查看。  

### 5.任务模型裁剪训练

此步骤可选，模型裁剪训练相对比普通的任务模型训练，需要消耗更多的时间，需要在正常任务模型训练的基础上，增加『**模型裁剪分类**』和『**模型裁剪训练**』两个步骤。  

裁剪过程将对模型各卷积层的敏感度信息进行分析，根据各参数对模型效果的影响进行不同比例的裁剪，再进行精调训练获得最终裁剪后的模型。  
裁剪训练后的模型体积，计算量都会减少，并且可以提升模型在低性能设备的预测速度，如移动端，边缘设备，CPU。

在可视化客户端上，**用户训练好模型后**，在训练界面，
- 首先，点击『模型裁剪分析』，此过程将会消耗较长的时间
- 接着，点击『开始模型裁剪训练』，客户端会创建一个新的任务，无需修改参数，直接再启动训练即可

<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880459-40cb4eeb-ce8e-40b3-8e75-7dda4544b116.jpg" width="800" />
</p>

### 6.模型效果评估

在模型评估页面，您可查看训练后的模型效果。评估方法包括混淆矩阵、精度、召回率等。

<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880511-5ff88ea7-69e2-4b88-bb27-13fc32268991.jpg" width="800" />
</p>

您还可以选择『数据集切分』时留出的『测试数据集』或从本地文件夹中导入一张/多张图片，将训练后的模型进行测试。根据测试结果，您可决定是否将训练完成的模型保存为预训练模型并进入模型发布页面，或返回先前步骤调整参数配置重新进行训练。

<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880513-385923a4-4abf-41b2-a97f-06c757c36ccf.jpg" width="800" />
</p>

### 7.模型发布

当模型效果满意后，您可根据实际的生产环境需求，选择将模型发布为需要的版本。  
如若要部署到移动端/边缘设备，对于部分支持量化的模型，还可以根据需求选择是否量化。量化可以压缩模型体积，提升预测速度

<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133880541-680b0db0-5b30-4806-8c1a-0eb05a68c70b.jpg" width="800" />
</p>
