# PaddleX Deployment部署方式

PaddleX Deployment适配业界常用的CPU、GPU（包括NVIDIA Jetson）、树莓派等硬件，支持PaddleClas PaddleDetection PaddleSeg三个套件的训练的部署，支持用户采用OpenVINO或TensorRT进行推理加速。完备支持工业最常使用的Windows系统，且提供C#语言进行部署的方式！
## 模型套件支持
本目录下代码，目前支持以下飞桨官方套件基于PaddleInference的部署。用户可参考[文件夹结构](./file_format.md)了解模型导出前后文件夹状态。

| 套件名称 | 版本号   | 支持模型 |
| -------- | -------- | ------- |
| PaddleDetection  | [release/2.2](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2)、[release/0.5](https://github.com/PaddlePaddle/PaddleDetection/tree/release/0.5) |  FasterRCNN / MaskRCNN / PPYOLO / PPYOLOv2 / YOLOv3   |  
| PaddleSeg        | [release/2.2](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2)       |  全部分割模型  |
| PaddleClas       | [release/2.2](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.2)      |  全部分类模型  |
| PaddleX          | [release/2.0.0](https://github.com/PaddlePaddle/PaddleX)                        |  全部静态图、动态图模型   |

## 硬件支持
- CPU(linux/windows)
- GPU(linux/windows)

## 各套件部署方式说明

- [PaddleX部署指南](./models/paddlex.md)
- [PaddleDetection部署指南](./models/paddledetection.md)
- [PaddleSeg部署指南](./models/paddleseg.md)
- [PaddleClas部署指南](./models/paddleclas.md)

## 模型加密与预测加速

- [模型加密预测示例](./demo/decrypt_infer.md)
- [PaddleInference集成TensorRT加载模型预测示例](./demo/tensorrt_infer.md)

## <h2 id="1">C++代码预测说明</h2>

- [部署相关API说明](./apis/model.md)
- [模型配置文件说明](./apis/yaml.md)
