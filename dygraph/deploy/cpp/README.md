## PaddlePaddle模型C++部署

本目录下代码，目前支持以下飞桨官方套件基于PaddleInference的部署。

还支持对ONNX进行部署。套件模型转换ONNX模型，参考[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)。

## 模型套件支持
- PaddleDetection([release/2.0](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.0))
- PaddleSeg([release/2.0](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v2.0))
- PaddleClas([release/2.1](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1))
- PaddleX([release/2.0-rc](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc))

## 硬件支持
- CPU(linux/windows)
- GPU(linux/windows)
- Jetson(TX2/Nano/Xavier)

## ONNX模型部署

### [Triton部署](./docs/compile/triton/docker.md)

Triton的全称为Triton Inference Server，由NVIDIA推出的具有低延迟、高吞吐等特性的高性能推理解决方案。它提供了针对CPU和GPU优化的云和边缘推理解决方案。 Triton支持HTTP / REST和GRPC协议，该协议允许远程客户端请求服务器管理的任何模型进行推理

### [TensorRT部署](./docs/compile/tensorrt/trt.md)

TensorRT是一个高性能的深度学习推理优化器，可以为深度学习应用提供低延迟、高吞吐率的部署推理。TensorRT核心是一个C++库，从 TensorRT 3 开始提供C++ API和Python API，主要用来针对 NVIDIA GPU进行 高性能推理（Inference）加速。

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
