# PaddleX全面升级动态图，2.0.0-rc发布！



<p align="center">
  <img src="../docs/gui/images/paddlex.png" width="360" height ="55" alt="PaddleX" align="middle" />
</p>
 <p align= "center"> PaddleX -- 飞桨全流程开发工具，以低代码的形式支持开发者快速实现产业实际项目落地 </p>

## :heart:重磅功能升级
* 全新发布Manufacture SDK，提供工业级多端多平台部署加速的预编译飞桨部署开发包（SDK），通过配置业务逻辑流程文件即可以低代码方式快速完成推理部署[欢迎体验](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/deploy/cpp)。

* PaddleX部署全面升级，支持飞桨视觉套件PaddleDetection、PaddleClas、PaddleSeg、PaddleX的统一部署能力。[欢迎体验](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/deploy/cpp)。



[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE) [![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleX.svg)](https://github.com/PaddlePaddle/PaddleX/releases) ![python version](https://img.shields.io/badge/python-3.6+-orange.svg) ![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
 ![QQGroup](https://img.shields.io/badge/QQ_Group-1045148026-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)

:hugs: PaddleX 集成飞桨智能视觉领域**图像分类**、**目标检测**、**语义分割**、**实例分割**任务能力，将深度学习开发全流程从**数据准备**、**模型训练与优化**到**多端部署**端到端打通，并提供**统一任务API接口**及**图形化开发界面Demo**。开发者无需分别安装不同套件，以**低代码**的形式即可快速完成飞桨全流程开发。

:factory: **PaddleX** 经过**质检**、**安防**、**巡检**、**遥感**、**零售**、**医疗**等十多个行业实际应用场景验证，沉淀产业实际经验，**并提供丰富的案例实践教程**，全程助力开发者产业实践落地。



:heart:**您可以前往  [完整PaddleX在线使用文档目录](https://paddlex.readthedocs.io/zh_CN/develop/index.html)  查看完整*Read the Doc* 格式的文档，获得更好的阅读体验**:heart:



![](../docs/gui/images/paddlexoverview.png)



## 安装

**PaddleX提供三种开发模式，满足用户的不同需求：**

1. **Python开发模式：**

   通过简洁易懂的Python API，在兼顾功能全面性、开发灵活性、集成方便性的基础上，给开发者最流畅的深度学习开发体验。<br>

  **前置依赖**
> - paddlepaddle == 2.1.0
> - 安装PaddlePaddle Develop版本，具体PaddlePaddle[安装主页](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/windows-pip.html)

**安装方式**

> - git clone --recurse-submodules https://github.com/PaddlePaddle/PaddleX.git
> - cd PaddleX/dygraph
> - pip install -r requirements.txt
> - pip install -r submodules.txt
> - python setup.py install


**特别说明**   Windows除了执行上述命令外，还需要下载pycocotools

> - pip install cython
> - pip install git+https://gitee.com/jiangjiajun/philferriere-cocoapi.git#subdirectory=PythonAPI


2. **Padlde GUI模式：**

   无代码开发的可视化客户端，应用Paddle API实现，使开发者快速进行产业项目验证，并为用户开发自有深度学习软件/应用提供参照。

- 前往[PaddleX官网](https://www.paddlepaddle.org.cn/paddle/paddlex)，申请下载PaddleX GUI一键绿色安装包。

- 前往[PaddleX GUI使用教程](../docs/gui/how_to_use.md)了解PaddleX GUI使用详情。

- [PaddleX GUI安装环境说明](../docs/gui/download.md)

3. **PaddleX Restful:**  
  使用基于RESTful API开发的GUI与Web Demo实现远程的深度学习全流程开发；同时开发者也可以基于RESTful API开发个性化的可视化界面
- 前往[PaddleX RESTful API使用教程](../docs/Resful_API/docs/readme.md)  


## 使用教程

1. **API模式：**

- [模型训练](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train)
- [模型剪裁](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/slim/prune)
- [模型导出]()


2. **GUI模式：**

- [图像分类](https://www.bilibili.com/video/BV1nK411F7J9?from=search&seid=3068181839691103009)
- [目标检测](https://www.bilibili.com/video/BV1HB4y1A73b?from=search&seid=3068181839691103009)
- [实例分割](https://www.bilibili.com/video/BV1M44y1r7s6?from=search&seid=3068181839691103009)
- [语义分割](https://www.bilibili.com/video/BV1qQ4y1Z7co?from=search&seid=3068181839691103009)

3. **模型部署：**
- [Manufacture SDK](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/deploy/cpp)
提供工业级多端多平台部署加速的预编译飞桨部署开发包（SDK），通过配置业务逻辑流程文件即可以低代码方式快速完成推理部署
- [PaddleX Deploy](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/deploy/cpp) 支持飞桨视觉套件PaddleDetection、PaddleClas、PaddleSeg、PaddleX的统一部署能力
## 产业级应用示例

- [钢筋计数](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/examples/rebar_count)
- [缺陷检测](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/examples/defect_detection)
- [机械手抓取](https://github.com/PaddlePaddle/PaddleX/tree/develop/dygraph/examples/robot_grab)
- [表计检测]()
## :question:[FAQ]
(../docs/gui/faq.md):question:

## 交流与反馈

- 项目官网：https://www.paddlepaddle.org.cn/paddle/paddlex

- PaddleX用户交流群：957286141 (手机QQ扫描如下二维码快速加入)  

  <p align="center">
    <img src="../docs/gui/images/QR2.jpg" width="250" height ="360" alt="QR" align="middle" />
  </p>



## 更新日志

- **2021.05.19 v2.0.0-rc**

  * 全面支持飞桨2.0动态图，更易用的开发模式
  * 目标检测任务新增[PP-YOLOv2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolov2.py), COCO test数据集精度达到49.5%、V100预测速度达到68.9 FPS
  * 目标检测任务新增4.2MB的超轻量级模型[PP-YOLO tiny](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolotiny.py)
  * 语义分割任务新增实时分割模型[BiSeNetV2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/semantic_segmentation/bisenetv2.py)
  * C++部署模块全面升级
    * PaddleInference部署适配2.0预测库[（使用文档）](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/deploy/cpp)
    * 支持飞桨[PaddleDetection]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddledetection.md)、[PaddleSeg]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleseg.md)、[PaddleClas](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleclas.md)以及PaddleX的模型部署
    * 新增基于PaddleInference的GPU多卡预测[（使用文档）](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/demo/multi_gpu_model_infer.md)
    * GPU部署新增基于ONNX的的TensorRT高性能加速引擎部署方式[（使用文档）]()
    * GPU部署新增基于ONNX的Triton服务化部署方式[（使用文档）](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/compile/triton/docker.md)


- **2020.09.07 v1.2.0**

  新增产业最实用目标检测模型PP-YOLO，FasterRCNN、MaskRCNN、YOLOv3、DeepLabv3p等模型新增内置COCO数据集预训练模型，适用于小模型精调。新增多种Backbone，优化体积及预测速度。优化OpenVINO、PaddleLite Android、服务端C++预测部署方案，新增树莓派部署方案等。

- **2020.07.12 v1.1.0**

  新增人像分割、工业标记读数案例。模型新增HRNet、FastSCNN、FasterRCNN，实例分割MaskRCNN新增Backbone HRNet。集成X2Paddle，PaddleX所有分类模型和语义分割模型支持导出为ONNX协议。新增模型加密Windows平台支持。新增Jetson、Paddle Lite模型部署预测方案。

- **2020.05.20 v1.0.0**

  新增C++和Python部署，模型加密部署，分类模型OpenVINO部署。新增模型可解释性接口

- **2020.05.17 v0.1.8**

  新增EasyData平台数据标注格式，支持imgaug数据增强库的pixel-level算子

## :hugs: 贡献代码:hugs:

我们非常欢迎您为PaddleX贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests。

### 开发者贡献项目

* [工业相机实时目标检测GUI](https://github.com/xmy0916/SoftwareofIndustrialCameraUsePaddle)
（windows系统，基于pyqt5开发）
* [工业相机实时目标检测GUI](https://github.com/LiKangyuLKY/PaddleXCsharp)
（windows系统，基于C#开发）
