<p align="center">
  <img src="./docs/images/paddlex.png" width="360" height ="60" alt="PaddleX" align="middle" />
</p>

[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleX.svg)](https://github.com/PaddlePaddle/PaddleX/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

PaddleX是基于飞桨核心框架、开发套件和工具组件的深度学习全流程开发工具。具备**全流程打通**、**融合产业实践**、**易用易集成**三大特点。

## 特点

- **全流程打通** 
  - 数据准备：支持LabelMe，精灵标注等主流数据标注工具协议，同时无缝集成[EasyData智能数据服务平台](https://ai.baidu.com/easydata/), 助力开发者高效获取AI开发所需高质量数据。
  - 模型训练：基于飞桨核心框架集成[PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)视觉开发套件，[VisualDL](https://github.com/PaddlePaddle/VisualDL)可视化分析组件，高效完成模型训练。
  - 多端部署：内置[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)模型压缩工具和AES模型加密SDK，结合Paddle Inference和[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)便捷完成高性能且可靠的多端部署。

- **融合产业实践** 
  - 精选飞桨产业实践的成熟模型结构，开放案例实践教程，加速开发者产业落地。
  - 通过[PaddleHub](https://github.com/PaddlePaddle/Paddle)内置丰富的飞桨高质量预训练模型，助力开发者高效实现飞桨Master模式。

- **易用易集成**
  - PadldeX提供简洁易用的全流程API，几行代码即可实现上百种数据增强、模型可解释性、C++模型部署等功能。
  - 提供以PaddleX API为核心集成的跨平台GUI界面，降低深度学习全流程应用门槛。


## 安装

PaddleX提供两种开发模式，满足不同场景和用户需求：

- **Python开发模式：** 通过Python API方式完成全流程使用或集成，该模型提供全面、灵活、开放的深度学习功能，有更高的定制化空间。

- **GUI开发模式：** 以PaddleX API为核心集成的跨平台GUI客户端，支持`Python开发模式`下的常用功能，以更低门槛的方式快速完成产业验证的模型训练。

开发者可根据自身需要按需选择不同的模式进行安装使用。


### Python开发模式安装

**前置依赖**
* paddlepaddle >= 1.8.0
* python >= 3.5
* cython
* pycocotools

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

### GUI开发模式安装

进入PaddleX官网[下载使用](https://www.paddlepaddle.org.cn/paddle/paddlex)，申请下载绿色安装包，开箱即用。
GUI模式的使用教程可参考[PaddleX GUI模式使用教程](https://paddlex.readthedocs.io/zh_CN/latest/paddlex_gui/index.html)

## 使用文档 

推荐访问[PaddleX在线使用文档](https://paddlex.readthedocs.io/zh_CN/latest/index.html)，快速查阅读使用教程和API文档说明。

- [10分钟快速上手](https://paddlex.readthedocs.io/zh_CN/latest/quick_start.html)
- [PaddleX模型训练](https://paddlex.readthedocs.io/zh_CN/latest/tutorials/train/index.html#id1)
- [PaddleX模型压缩](https://paddlex.readthedocs.io/zh_CN/latest/slim/index.html#id1)
- [PaddleX模型库](https://paddlex.readthedocs.io/zh_CN/latest/model_zoo.html#id1)
- [PaddleX多端部署](docs/deploy.md)

## 在线教程

基于AIStudio平台，快速在线体验PaddleX的Python开发模式教程。

- [PaddleX快速上手——MobileNetV3-ssld 化妆品分类](https://aistudio.baidu.com/aistudio/projectdetail/450220)
- [PaddleX快速上手——Faster-RCNN AI识虫](https://aistudio.baidu.com/aistudio/projectdetail/439888)
- [PaddleX快速上手——DeepLabv3+ 视盘分割](https://aistudio.baidu.com/aistudio/projectdetail/440197)

## 交流与反馈

- 项目官网: https://www.paddlepaddle.org.cn/paddle/paddlex
- PaddleX用户QQ群: 1045148026 (手机QQ扫描如下二维码快速加入)  
<img src="./docs/images/QQGroup.jpeg" width="195" height="300" alt="QQGroup" align="center" />

## FAQ

## 更新日志

## 贡献代码

我们非常欢迎您为PaddleX贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests.
