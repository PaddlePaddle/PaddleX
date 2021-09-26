## Deployment部署方式

## 目录
- [模型套件支持](#模型套件支持)
- [硬件支持](#硬件支持)
- [具体步骤](#具体步骤)
  - [1.PaddleInference编译说明](#1PaddleInference编译说明)
  - [2.部署模型导出](#2部署模型导出)
  - [3.模型预测](#3模型预测)
- [模型加密与预测加速](#模型加密与预测加速)
- [CPP代码预测说明](#CPP代码预测说明)


## 模型套件支持
本目录下代码，目前支持以下飞桨官方套件基于PaddleInference的部署。

| 套件名称 | 版本号   | 支持模型 | 
| -------- | -------- | ------- |
| PaddleDetection  | [release/2.2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2)、[release/0.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.5) |  FasterRCNN / MaskRCNN / PPYOLO / PPYOLOv2 / YOLOv3   |  
| PaddleSeg        | [release/2.2](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2)       |  全部  |
| PaddleClas       | [release/2.2](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.2)      |  全部  |
| PaddleX          | [release/2.0.0](https://github.com/PaddlePaddle/PaddleX)                        |  静态图、动态图   |

## 硬件支持
- CPU(linux/windows)
- GPU(linux/windows)

## 各套件部署

- [PaddleX部署指南](./docs/models/paddlex.md/#部署模型导出)
- [PaddleDetection部署指南](./docs/models/paddledetection.md/#部署模型导出)
- [PaddleSeg部署指南](./docs/models/paddleseg.md/#部署模型导出)
- [PaddleClas部署指南](./docs/models/paddleclas.md/#部署模型导出)

## 模型加密

- [模型加密预测示例](./docs/demo/decrypt_infer.md)

## CPP代码预测说明

- [部署相关API说明](./docs/apis/model.md)
- [模型配置文件说明](./docs/apis/yaml.md)
