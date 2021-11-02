# 部署方式概览

PaddleX提供了多种部署方式，用户可根据实际需要选择本地部署、边缘侧部署、服务化部署、Docker部署。部署方式目录如下：

  - 本地部署
    - C++部署
      - [C++源码编译](./../deploy/cpp/README.md)
      - [C#工程化示例](./../deploy/cpp/docs/CSharp_deploy)
    - [Python部署](./../docs/python_deploy.md)
  - 边缘侧部署
    - [NVIDIA-Jetson部署(C++)](./../deploy/cpp/docs/compile/paddle/jetson.md)

  - 服务化部署
    - [HubServing部署（Python）](./../docs/hub_serving_deploy.md)
  - [基于ONNX部署（C++）](./../deploy/cpp/docs/compile/README.md)
    - [OpenVINO推理引擎](./../deploy/cpp/docs/compile/openvino/README.md)
    - [Triton部署](./../deploy/cpp/docs/compile/triton/docker.md)
