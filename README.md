<img src="./paddlex.png" width = "300" height = "47" alt="PaddleX" align=center />

[![Build Status](https://travis-ci.org/PaddlePaddle/PaddleX.svg?branch=release/v1.6)](https://travis-ci.org/PaddlePaddle/PaddleX)
[![License](https://img.shields.io/badge/license-Apache%202-red.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleX.svg)](https://github.com/PaddlePaddle/PaddleX/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

PaddleX是基于飞桨开发套件和工具组件的全流程深度学习开发工具。具备易集成，易使用，全流程等特点。PaddleX作为深度学习开发工具，不仅提供了开源的内核代码，可供用户灵活使用或集成，同时也提供了配套的前端可视化客户端套件，让用户以可视化的方式进行模型开发，免去代码开发过程。

访问[PaddleX官网](https://www.paddlepaddle.org.cn/paddle/paddlex)获取更多细节。

## 快速安装

PaddleX提供两种使用模式，满足不同的场景和用户需求：
- **开发模式：** pip安装PaddleX后，开发者可通过Python API调用方式更灵活地完成模型的训练或软件集成。
- **可视化模式：** 通过绿色安装的跨平台软件包，用户即可开箱即用，以可视化方式快速体验飞桨深度学习的全流程。

### 开发模式

**前置依赖**
* python >= 3.5
* cython
* pycocotools

```
pip install paddlex -i https://mirror.baidu.com/pypi/simple
```

### 可视化模式

进入PaddleX官网[下载使用](https://www.paddlepaddle.org.cn/paddle/paddlex)，申请下载绿色安装包，开箱即用。

## 文档

推荐访问[PaddleX在线使用文档](https://paddlex.readthedocs.io/zh_CN/latest/index.html)，快速查阅读使用教程和API文档说明。

- [10分钟快速上手PaddleX模型训练](docs/quick_start.md)
- [PaddleX使用教程](docs/tutorials)
- [PaddleX模型库](docs/model_zoo.md)
- [模型多端部署](docs/deploy.md)
- [PaddleX可视化模式进行模型训练](docs/client_use.md)


## 反馈

- 项目官网: https://www.paddlepaddle.org.cn/paddle/paddlex
- PaddleX用户QQ群: 1045148026 (手机QQ扫描如下二维码快速加入)  
<img src="./QQGroup.jpeg" width="195" height="300" alt="QQGroup" align=center />


## 飞桨技术生态

PaddleX全流程开发工具依赖以下飞桨开发套件与工具组件

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas)
- [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)
- [PaddleHub](https://github.com/PaddlePaddle/PaddleHub)
- [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)
- [VisualDL](https://github.com/PaddlePaddle/VisualDL)
