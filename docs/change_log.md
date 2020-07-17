# 更新日志


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
> - [人像分割案例]()
> - [工业表计读数案例]()
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
