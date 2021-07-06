# 模型部署

我们已发布全新的cpp deploy部署模块，除了支持PaddleX静态图、动态图导出模型，还支持PaddleDetection、PaddleSeg、PaddleClas套件导出的模型进行部署。同时在新模块还支持TensorRT、Triton等第三方推理引擎，详情请看文档：

- [全新cpp部署](../dygraph/deploy/cpp)


本目录为PaddleX模型静态图版本部署代码，编译和使用教程参考：

- [服务端部署](../docs/deploy/server/)
  - [Python部署](../docs/deploy/server/python.md)
  - [C++部署](../docs/deploy/server/cpp/)
    - [Windows平台部署](../docs/deploy/server/cpp/windows.md)
    - [Linux平台部署](../docs/deploy/server/cpp/linux.md)
  - [模型加密部署](../docs/deploy/server/encryption.md)
- [Nvidia Jetson开发板部署](../docs/deploy/nvidia-jetson.md)
- [移动端部署](../docs/deploy/paddlelite/)
  - [模型压缩](../docs/deploy/paddlelite/slim)
    - [模型量化](../docs/deploy/paddlelite/slim/quant.md)
    - [模型裁剪](../docs/deploy/paddlelite/slim/prune.md)
  - [Android平台](../docs/deploy/paddlelite/android.md)
- [OpenVINO部署](../docs/deploy/openvino/introduction.md)
- [树莓派部署](../docs/deploy/raspberry/Raspberry.md)
