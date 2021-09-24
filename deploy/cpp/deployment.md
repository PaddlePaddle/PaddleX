## Deployment部署方式

## 目录
- [模型套件支持](#模型套件支持)
- [硬件支持](#硬件支持)
- [具体步骤](#具体步骤)
  - [1.PaddleInference编译说明](#1PaddleInference编译说明)
  - [2.部署模型导出](#2部署模型导出)
  - [3.模型预测](#3模型预测)
- [模型加密与预测加速](#模型加密与预测加速)
- [API说明](#API说明)


## 模型套件支持
本目录下代码，目前支持以下飞桨官方套件基于PaddleInference的部署。
- PaddleDetection([release/2.2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2))：当前支持PaddleDetection release/0.5和release/2.2分支导出的部分模型进行导出及部署，支持FasterRCNN / MaskRCNN / PPYOLO / PPYOLOv2 / YOLOv3。
- PaddleSeg([release/2.2](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2))：当前支持PaddleSeg release/2.2分支训练的模型进行导出及部署。
- PaddleClas([release/2.2](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.2))：当前支持PaddleClas release/2.2分支导出的模型进行部署。
- PaddleX([release/2.0.0](https://github.com/PaddlePaddle/PaddleX))：当前对PaddleX静态图和动态图版本导出的模型都支持

## 硬件支持
- CPU(linux/windows)
- GPU(linux/windows)

## 具体步骤
### 1.PaddleInference编译说明
- [Linux编译(支持加密)指南](./docs/compile/paddle/linux.md)
- [Windows编译(支持加密)指南](./docs/compile/paddle/windows.md)

### 2.部署模型导出
- [PaddleX部署模型导出](./docs/models/paddlex.md/#部署模型导出)
- [PaddleDetection部署模型导出](./docs/models/paddledetection.md/#部署模型导出)
- [PaddleSeg部署模型导出](./docs/models/paddleseg.md/#部署模型导出)
- [PaddleClas部署模型导出](./docs/models/paddleclas.md/#部署模型导出)

### 3.模型预测
- [PaddleX模型预测](./docs/models/paddlex.md/#模型预测)
- [PaddleDetection模型预测](./docs/models/paddledetection.md/#模型预测)
- [PaddleSeg模型预测](./docs/models/paddleseg.md/#模型预测)
- [PaddleClas模型预测](./docs/models/paddleclas.md/#模型预测)
- [模型加载预测示例](./docs/demo/model_infer.md)
- [参数说明](./docs/demo/model_infer.md/#参数说明)

## 模型加密与预测加速

- [模型加密预测示例](./docs/demo/decrypt_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](./docs/demo/tensorrt_infer.md)

## API说明

- [部署相关API说明](./docs/apis/model.md)
- [模型配置文件说明](./docs/apis/yaml.md)

