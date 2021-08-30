# 版本更新信息

## 最新版本信息

- **2021.07.06 v2.0.0-rc3**

  * PaddleX部署全面升级，支持飞桨视觉套件PaddleDetection、PaddleClas、PaddleSeg、PaddleX的端到端统一部署能力。[使用教程](https://github.com/PaddlePaddle/PaddleX/tree/develop/deploy/cpp)
  * 全新发布Manufacture SDK，提供工业级多端多平台部署加速的预编译飞桨部署开发包（SDK），通过配置业务逻辑流程文件即可以低代码方式快速完成推理部署。[使用教程](https://github.com/PaddlePaddle/PaddleX/tree/develop/deploy/cpp/docs/manufacture_sdk)
  * 发布产业实践案例：钢筋计数、缺陷检测、机械手抓取、工业表计读数、Windows系统下使用C#语言部署。[使用教程](https://github.com/PaddlePaddle/PaddleX/tree/develop/examples)
  * 升级PaddleX GUI，支持30系列显卡、新增模型PP-YOLO V2、PP-YOLO Tiny 、BiSeNetV2。[使用教程](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/install.md#2-padldex-gui%E5%BC%80%E5%8F%91%E6%A8%A1%E5%BC%8F%E5%AE%89%E8%A3%85)


- **2021.05.19 v2.0.0-rc**

  * 全面支持飞桨2.0动态图，更易用的开发模式
  * 目标检测任务新增[PP-YOLOv2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolov2.py), COCO test数据集精度达到49.5%、V100预测速度达到68.9 FPS
  * 目标检测任务新增4.2MB的超轻量级模型[PP-YOLO tiny](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolotiny.py)
  * 语义分割任务新增实时分割模型[BiSeNetV2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/semantic_segmentation/bisenetv2.py)
  * C++部署模块全面升级
    * PaddleInference部署适配2.0预测库[（使用文档）](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/deploy/cpp)
    * 支持飞桨[PaddleDetection]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddledetection.md)、[PaddleSeg]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleseg.md)、[PaddleClas](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleclas.md)以及PaddleX的模型部署
    * 新增基于PaddleInference的GPU多卡预测[（使用文档）](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/demo/multi_gpu_model_infer.md)
    * GPU部署新增基于ONNX的的TensorRT高性能加速引擎部署方式[（使用文档）]()
    * GPU部署新增基于ONNX的Triton服务化部署方式[（使用文档）](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/compile/triton/docker.md)


## 历史版本信息

- **2020.09.07 v1.2.0**

  新增产业最实用目标检测模型PP-YOLO，FasterRCNN、MaskRCNN、YOLOv3、DeepLabv3p等模型新增内置COCO数据集预训练模型，适用于小模型精调。新增多种Backbone，优化体积及预测速度。优化OpenVINO、PaddleLite Android、服务端C++预测部署方案，新增树莓派部署方案等。

- **2020.07.12 v1.1.0**

  新增人像分割、工业标记读数案例。模型新增HRNet、FastSCNN、FasterRCNN，实例分割MaskRCNN新增Backbone HRNet。集成X2Paddle，PaddleX所有分类模型和语义分割模型支持导出为ONNX协议。新增模型加密Windows平台支持。新增Jetson、Paddle Lite模型部署预测方案。

- **2020.05.20 v1.0.0**

  新增C++和Python部署，模型加密部署，分类模型OpenVINO部署。新增模型可解释性接口

- **2020.05.17 v0.1.8**

  新增EasyData平台数据标注格式，支持imgaug数据增强库的pixel-level算子
