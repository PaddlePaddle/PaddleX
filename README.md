<p align="center">
  <img src="./docs/gui/images/paddlex.png" width="360" height ="55" alt="PaddleX" align="middle" />
</p>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleX.svg)](https://github.com/PaddlePaddle/PaddleX/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![QQGroup](https://img.shields.io/badge/QQ_Group-1045148026-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20)

**PaddleX--飞桨全功能开发套件**，集成了飞桨视觉套件（PaddleClas、PaddleDetection、PaddleSeg）、模型压缩工具PaddleSlim、可视化分析工具VisualDL、轻量化推理引擎Paddle Lite 等核心模块的能力，同时融合飞桨团队丰富的实际经验及技术积累，将深度学习开发全流程，从数据准备、模型训练与优化到多端部署实现了端到端打通，为开发者提供飞桨全流程开发的最佳实践。

**PaddleX 提供了最简化的API设计，并官方实现GUI供大家下载使用**，最大程度降低开发者使用门槛。开发者既可以应用**PaddleX GUI**快速体验深度学习模型开发的全流程，也可以直接使用 **PaddleX API** 更灵活地进行开发。

更进一步的，如果用户需要根据自己场景及需求，定制化地对PaddleX 进行改造或集成，PaddleX 也提供很好的支持。

## PaddleX 三大特点

### 全流程打通

- **数据准备**：兼容ImageNet、VOC、COCO等常用数据协议, 同时与Labelme、精灵标注助手、[EasyData智能数据服务平台](https://ai.baidu.com/easydata/)等无缝衔接，全方位助力开发者更快完成数据准备工作。

- **数据预处理及增强**：提供极简的图像预处理和增强方法--Transforms，适配imgaug图像增强库，支持上百种数据增强策略，是开发者快速缓解小样本数据训练的问题。

- **模型训练**：集成[PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)视觉开发套件，提供大量精选的、经过产业实践的高质量预训练模型，使开发者更快实现工业级模型效果。

- **模型调优**：内置模型可解释性模块、[VisualDL](https://github.com/PaddlePaddle/VisualDL)可视化分析工具。使开发者可以更直观的理解模型的特征提取区域、训练过程参数变化，从而快速优化模型。

- **多端安全部署**：内置[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)模型压缩工具和**模型加密部署模块**，与飞桨原生预测库Paddle Inference及高性能端侧推理引擎[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) 无缝打通，使开发者快速实现模型的多端、高性能、安全部署。

### 融合产业实践

- **产业验证**：经过**质检**、**安防**、**巡检**、**遥感**、**零售**、**医疗**等十多个行业实际应用场景验证，适配行业数据格式及部署环境要求。
- **经验沉淀**：沉淀产业实践实际经验，**提供丰富的案例实践教程**，加速开发者产业落地。
- **产业开发者共建**：吸收实际产业开发者贡献代码，源于产业，回馈产业。



## 易用易集成

- **易用**：统一的全流程API，5步即可完成模型训练，10行代码实现Python/C++高性能部署。
- **易集成**：支持开发者自主改造、集成，开发出适用于自己产业需求的产品。并官方提供基于 PaddleX API 开发的跨平台可视化工具-- **PaddleX GUI**，使开发者快速体验飞桨深度学习开发全流程，并启发用户进行定制化开发。

## 安装

**PaddleX提供两种开发模式，满足用户的不同需求：**

1. **Python开发模式：** 通过简洁易懂的Python API，在兼顾功能全面性、开发灵活性、集成方便性的基础上，给开发者最流畅的深度学习开发体验。
**前置依赖**
* paddlepaddle >= 1.8.0
* python >= 3.5
* cython
* pycocotools

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```
详细安装方法请参考[PaddleX安装](https://paddlex.readthedocs.io/zh_CN/develop/install.html)


2.  **Padlde GUI模式：** 无代码开发的可视化客户端，应用Paddle API实现，使开发者快速进行产业项目验证，并为用户开发自有深度学习软件/应用提供参照。

您可前往[PaddleX官网](https://www.paddlepaddle.org.cn/paddle/paddlex)，申请下载Paddle X GUI一键绿色安装包。

您可前往[PaddleX GUI使用教程](./docs/gui/how_to_use.md)了解PaddleX GUI使用详情。


## 完整使用文档及API说明

[PaddleX在线使用文档](https://paddlex.readthedocs.io/zh_CN/develop/index.html)。

- [10分钟快速上手使用](https://paddlex.readthedocs.io/zh_CN/develop/quick_start.html)
- [PaddleX模型训练教程集合](https://paddlex.readthedocs.io/zh_CN/develop/train/index.html)
- [PaddleX API参考文档](https://paddlex.readthedocs.io/zh_CN/develop/apis/index.html)

## 在线项目示例

为了使开发者更快掌握PaddleX API，我们创建了一系列完整的示例教程，您可通过AIStudio一站式开发平台，快速在线运行PaddleX的项目。

- [PaddleX快速上手CV模型训练](https://aistudio.baidu.com/aistudio/projectdetail/450925)
- [PaddleX快速上手——MobileNetV3-ssld 化妆品分类](https://aistudio.baidu.com/aistudio/projectdetail/450220)
- [PaddleX快速上手——Faster-RCNN AI识虫](https://aistudio.baidu.com/aistudio/projectdetail/439888)
- [PaddleX快速上手——DeepLabv3+ 视盘分割](https://aistudio.baidu.com/aistudio/projectdetail/440197)

## 交流与反馈

- 项目官网: https://www.paddlepaddle.org.cn/paddle/paddlex
- PaddleX用户交流群: 1045148026 (手机QQ扫描如下二维码快速加入)  
<img src="./docs/gui/images/QR.jpg" width="250" height="300" alt="QQGroup" align="center" />

## [FAQ](./docs/gui/faq.md)

## 更新日志
> [历史版本及更新内容](https://paddlex.readthedocs.io/zh_CN/develop/change_log.html)

- 2020.07.12 v1.0.8
- 2020.05.20 v1.0.0
- 2020.05.17 v0.1.8

## 贡献代码

我们非常欢迎您为PaddleX贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests.
