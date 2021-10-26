# 部署方式概览

PaddleX提供了多种部署方式，用户可根据实际需要选择本地部署、边缘侧部署、服务化部署、Docker部署。部署方式目录如下：

- [部署方式概览](./deploy)
  - 本地部署
    - [OpenVINO](./deploy/cpp/docs/compile/openvino/README.md)（C++）
    - [C++部署](./deploy/cpp)
      - [Manufacture SDK](./deploy/cpp/docs/manufacture_sdk)
      - [Deployment SDK](./deploy/cpp/docs/deployment.md)
      - [C#工程化部署](./deploy/cpp/docs/C#_deploy)
    - [Python部署](./docs/python_deploy.md)
  - 边缘侧部署
    - [NVIDIA-JetsonQT部署](./deploy/cpp/docs/jetson-deploy)
  - 服务化部署
    - [HubServing部署](./docs/hub_serving_deploy.md)
  - Docker部署(C++)
    - [Triton部署](./deploy/cpp/docs/compile/triton/docker.md)
    - [TensorRT部署](./deploy/cpp/docs/compile/tensorrt/trt.md)

- [模型加密](./deploy/cpp/docs/demo/decrypt_infer.md)
- [ONNX格式转换](./deploy/cpp/docs/compile)
