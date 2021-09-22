## PaddlePaddle模型C++部署

本目录下代码，目前支持以下飞桨官方套件基于PaddleInference的部署。

## 模型套件支持
- PaddleDetection([release/2.1](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1))
- PaddleSeg([release/2.1](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1))
- PaddleClas([release/2.1](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1))
- PaddleX([release/2.0-rc](https://github.com/PaddlePaddle/PaddleX))

## 硬件支持
- CPU(linux/windows)
- GPU(linux/windows)

## 文档
### PaddleInference编译说明
- [Linux编译(支持加密)指南](./docs/compile/paddle/linux.md)
- [Windows编译(支持加密)指南](./docs/compile/paddle/windows.md)
- [Jetson编译指南](./docs/compile/paddle/jetson.md)

### 模型部署说明
- [PaddleX部署指南](./docs/models/paddlex.md)
- [PaddleDetection部署指南](./docs/models/paddledetection.md)
- [PaddleSeg部署指南](./docs/models/paddleseg.md)
- [PaddleClas部署指南](./docs/models/paddleclas.md)

### 模型预测示例
- [单卡加载模型预测示例](./docs/demo/model_infer.md)
- [多卡加载模型预测示例](./docs/demo/multi_gpu_model_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](./docs/demo/tensorrt_infer.md)
- [模型加密预测示例](./docs/demo/decrypt_infer.md)

### API说明

- [部署相关API说明](./docs/apis/model.md)
- [模型配置文件说明](./docs/apis/yaml.md)

