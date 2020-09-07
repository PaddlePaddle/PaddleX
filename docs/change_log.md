# 更新日志

**v1.2.0** 2020.09.07
- 模型更新
  > - 新增目标检测模型PPYOLO[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo)
  > - FasterRCNN、MaskRCNN、YOLOv3、DeepLabv3p等模型新增内置COCO数据集预训练模型
  > - 目标检测模型FasterRCNN和MaskRCNN新增backbone HRNet_W18[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn)
  > - 语义分割模型DeepLabv3p新增backbone MobileNetV3_large_ssld[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p)

- 模型部署更新
  > - 新增模型通过OpenVINO的部署方案[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/deploy/openvino/index.html)
  > - 新增模型在树莓派上的部署方案[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/deploy/raspberry/index.html)
  > - 优化PaddleLite Android部署的数据预处理和后处理代码性能
  > - 优化Paddle服务端C++代码部署代码，增加use_mkl等参数，通过mkldnn显著提升模型在CPU上的预测性能

- 产业案例更新
  > - 新增RGB图像遥感分割案例[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/examples/remote_sensing.html)
  > - 新增多通道遥感分割案例[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/examples/multi-channel_remote_sensing/README.html)

- 其它
  > - 新增数据集切分功能，支持通过命令行切分ImageNet、PascalVOC、MSCOCO和语义分割数据集[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/data/format/classification.html#id2)

**v1.1.0** 2020.07.12

- 模型更新
> - 新增语义分割模型HRNet、FastSCNN
> - 目标检测FasterRCNN、实例分割MaskRCNN新增backbone HRNet
> - 目标检测/实例分割模型新增COCO数据集预训练模型
> - 集成X2Paddle，PaddleX所有分类模型和语义分割模型支持导出为ONNX协议
- 模型部署更新
> - 模型加密增加支持Windows平台
> - 新增Jetson、Paddle Lite模型部署预测方案
> - C++部署代码新增batch批预测，并采用OpenMP对预处理进行并行加速
- 新增2个PaddleX产业案例
> - [人像分割案例](https://paddlex.readthedocs.io/zh_CN/develop/examples/human_segmentation.html)
> - [工业表计读数案例](https://paddlex.readthedocs.io/zh_CN/develop/examples/meter_reader.html)
- 新增数据格式转换功能，LabelMe、精灵标注助手和EasyData平台标注的数据转为PaddleX支持加载的数据格式
- PaddleX文档更新，优化文档结构


**v1.0.0** 2020.05.20

- 增加模型C++部署和Python部署代码
- 增加模型加密部署方案
- 增加分类模型的OpenVINO部署方案
- 增加模型可解释性的接口


**v0.1.8** 2020.05.17

- 修复部分代码Bug
- 新增EasyData平台数据标注格式支持
- 支持imgaug数据增强库的pixel-level算子
