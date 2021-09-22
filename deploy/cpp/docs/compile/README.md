## ONNX模型部署
Paddle的模型除了直接通过PaddleInference部署外，还可以通过[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX.git)转为ONNX后使用第三方推理引擎进行部署，在本目录下，我们提供了基于OpenVINO、Triton和TensorRT三个引擎的部署支持。
- [OpenVINO部署](./docs/compile/openvino/README.md)
- [Triton部署](./docs/compile/triton/docker.md)
- [TensorRT部署](./docs/compile/tensorrt/trt.md)
