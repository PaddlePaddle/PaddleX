# 更新日志

**v1.3.0** 2020.12.20

- 模型更新
  > - 图像分类模型ResNet50_vd新增10万分类预训练模型 
  > - 目标检测模型FasterRCNN新增模型裁剪支持
  > - 目标检测模型新增多通道图像训练支持

- 模型部署更新
  > - 修复OpenVINO部署C++代码中部分Bug
  > - 树莓派部署新增Arm V8支持

- 产业案例更新
 > - 新增工业质检产业案例，提供基于GPU和CPU两种部署场景下的工业质检方案，及与质检相关的优化策略 [详情链接](https://paddlex.readthedocs.io/zh_CN/develop/examples/industrial_quality_inspection)

- **新增RestFUL API模块**
新增RestFUL API模块，开发者可通过此模块快速开发基于PaddleX的训练平台
 > - 增加基于RestFUL API的HTML Demo [详情链接](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/gui/introduction.md#paddlex-web-demo)
 > - 增加基于RestFUL API的Remote版可视化客户端 [详情链接](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/gui/introduction.md#paddlex-remote-gui)
新增模型通过OpenVINO的部署方案[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/deploy/openvino/index.html)

**v1.2.0** 2020.09.07
- 模型更新
  > - 新增产业最实用目标检测模型PP-YOLO，深入考虑产业应用对精度速度的双重面诉求，COCO数据集精度45.2%，Tesla V100预测速度72.9FPS。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-ppyolo)
  > - FasterRCNN、MaskRCNN、YOLOv3、DeepLabv3p等模型新增内置COCO数据集预训练模型，适用于小数据集的微调训练。
  > - 目标检测模型FasterRCNN和MaskRCNN新增backbone HRNet_W18，适用于对细节预测要求较高的应用场景。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn)
  > - 语义分割模型DeepLabv3p新增backbone MobileNetV3_large_ssld，模型体积9.3MB，Cityscapes数据集精度仍保持有73.28%。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p)

- 模型部署更新
  > - 新增模型通过OpenVINO预测加速的部署方案，CPU上相比mkldnn加速库预测速度提升1.5～2倍左右。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/deploy/openvino/index.html)
  > - 新增模型在树莓派上的部署方案，进一步丰富边缘侧的部署方案。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/deploy/raspberry/index.html)
  > - 优化PaddleLite Android部署的数据预处理和后处理代码性能，预处理速度提升10倍左右，后处理速度提升4倍左右。
  > - 优化Paddle服务端C++代码部署代码，增加use_mkl等参数，CPU上相比未开启mkldnn预测速度提升10～50倍左右。

- 产业案例更新
  > - 新增大尺寸RGB图像遥感分割案例，提供滑动窗口预测接口，不仅能避免显存不足的发生，而且能通过配置重叠程度消除最终预测结果中各窗口拼接处的裂痕感。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/examples/remote_sensing.html)
  > - 新增多通道遥感影像分割案例，打通语义分割任务对任意通道数量的数据分析、模型训练、模型部署全流程。[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/examples/multi-channel_remote_sensing/README.html)

- 其它
  > - 新增数据集切分功能，支持通过命令行一键切分ImageNet、PascalVOC、MSCOCO和语义分割数据集[详情链接](https://paddlex.readthedocs.io/zh_CN/develop/data/format/classification.html#id2)

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
