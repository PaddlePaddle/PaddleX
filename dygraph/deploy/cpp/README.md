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
- Jetson(TX2/Nano/Xavier)

## 文档
### PaddleInference编译说明
- [Linux编译指南](./docs/compile/paddle/linux.md)
- [Windows编译指南](./docs/compile/paddle/windows.md)
- [Jetson编译指南](./docs/compile/paddle/jetson.md)

### 模型部署说明
- [PaddleDetection部署指南](./docs/models/paddledetection.md)
- [PaddleSeg部署指南](./docs/models/paddleseg.md)
- [PaddleClas部署指南](./docs/models/paddleclas.md)

### API说明

- [部署相关API说明](./docs/apis/model.md)
- [模型配置文件说明](./docs/apis/yaml.md)


## ONNX模型部署
Paddle的模型除了直接通过PaddleInference部署外，还可以通过[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)转为ONNX后使用第三方推理引擎进行部署，在本目录下，我们提供了基于Triton和TensorRT两个引擎的部署支持。
- [Triton部署](./docs/compile/triton/docker.md)
- [TensorRT部署](./docs/compile/tensorrt/trt.md)
