## PaddlePaddle模型C++部署

本目录下代码，目前支持以下飞桨官方套件基于PaddleInference的部署。

## 模型套件支持
- PaddleDetection([release/2.1](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1))：当前支持PaddleDetection release/0.5和release/2.1分支导出的部分模型进行导出及部署，支持FasterRCNN / MaskRCNN / PPYOLO / PPYOLOv2 / YOLOv3。
- PaddleSeg([release/2.1](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1))：当前支持PaddleSeg release/2.1分支训练的模型进行导出及部署。
- PaddleClas([release/2.1](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1))：当前支持PaddleClas release/2.1分支导出的模型进行部署。
- PaddleX([release/2.0-rc](https://github.com/PaddlePaddle/PaddleX))：当前对PaddleX静态图和动态图版本导出的模型都支持

## 硬件支持
- CPU(linux/windows)
- GPU(linux/windows)

## 部署方案
### 步骤一、PaddleInference编译说明
- [Linux编译(支持加密)指南](./docs/compile/paddle/linux.md)
- [Windows编译(支持加密)指南](./docs/compile/paddle/windows.md)

### 步骤二、部署模型导出
- [PaddleX部署模型导出](./docs/models/paddlex.md/#部署模型导出)
- [PaddleDetection部署模型导出](./docs/models/paddledetection.md)
- [PaddleSeg部署模型导出](./docs/models/paddleseg.md)
- [PaddleClas部署模型导出](./docs/models/paddleclas.md)

### 步骤三、模型预测
- [PaddleX模型预测](./docs/models/paddlex.md)
- [PaddleDetection模型预测](./docs/models/paddledetection.md)
- [PaddleSeg模型预测](./docs/models/paddleseg.md)
- [PaddleClas模型预测](./docs/models/paddleclas.md)




### 模型预测示例
- [单卡加载模型预测示例](./docs/demo/model_infer.md)
- [多卡加载模型预测示例](./docs/demo/multi_gpu_model_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](./docs/demo/tensorrt_infer.md)
- [模型加密预测示例](./docs/demo/decrypt_infer.md)

### API说明

- [部署相关API说明](./docs/apis/model.md)
- [模型配置文件说明](./docs/apis/yaml.md)

