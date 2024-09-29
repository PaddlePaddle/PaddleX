# 版本更新信息

## 最新版本信息
### PaddleX v3.0.0beta1(9.30/2024)
PaddleX 3.0 Beta1 提供 200+ 模型通过极简的 Python API 一键调用；实现基于统一命令的模型全流程开发，并开源 PP-ChatOCRv3 特色模型产线基础能力；支持 100+ 模型高性能推理和服务化部署，7 类重点视觉模型端侧部署；70+ 模型开发全流程适配昇腾 910B，15+ 模型开发全流程适配昆仑芯和寒武纪。

- **模型丰富一键调用：** 将覆盖文档图像智能分析、OCR、目标检测、时序预测等多个关键领域的 200+ 飞桨模型整合为 13 条模型产线，通过极简的 Python API 一键调用，快速体验模型效果。同时支持 20+ 单功能模块，方便开发者进行模型组合使用。
- **提高效率降低门槛：** 实现基于图形界面和统一命令的模型全流程开发，打造大小模型结合、大模型半监督学习和多模型融合的8条特色模型产线，大幅度降低迭代模型的成本。
- **多种场景灵活部署：** 支持高性能部署、服务化部署和端侧部署等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。
- **主流硬件高效支持：** 支持英伟达 GPU、昆仑芯、昇腾和寒武纪等多种主流硬件的无缝切换，确保高效运行。

### PaddleX v3.0.0beta(6.27/2024)
PaddleX 3.0beta 集成了飞桨生态的优势能力，覆盖 7 大场景任务，构建了 16 条模型产线，提供低代码开发模式，助力开发者在多种主流硬件上实现模型全流程开发。

- **基础模型产线（模型丰富，场景全面）：** 精选 68 个优质飞桨模型，涵盖图像分类、目标检测、图像分割、OCR、文本图像版面分析、时序预测等任务场景。
- **特色模型产线（显著提升效率）：** 提供大小模型结合、大模型半监督学习和多模型融合的高效解决方案。
- **低代码开发模式（便捷开发与部署）：** 提供零代码和低代码两种开发方式。
  - 零代码开发：用户通过图形界面（GUI）交互式提交后台训练任务，打通在线和离线部署，并支持以 API 形式调用在线服务。
  - 低代码开发：通过统一的 API 接口实现 16 条模型产线的全流程开发，同时支持用户自定义模型流程串联。
- **多硬件本地支持（兼容性强）：** 支持英伟达 GPU、昆仑芯、昇腾和寒武纪等多种硬件，纯离线使用。

### PaddleX v2.1.0(12.10/2021)

新增超轻量分类模型PPLCNet，在Intel CPU上，单张图像预测速度约5ms，ImageNet-1K数据集上Top1识别准确率达到80.82%，超越ResNet152的模型效果 欢迎体验
新增轻量级检测特色模型PP-PicoDet，第一个在1M参数量之内mAP(0.5:0.95)超越30+(输入416像素时)，网络预测在ARM CPU下可达150FPS 欢迎体验
升级PaddleX Restful API，支持飞桨动态图开发模式 欢迎体验
新增检测模型负样本训练策略 欢迎体验
新增python轻量级服务化部署 欢迎体验

### PaddleX v2.0.0(9.10/2021)
* PaddleX API
  - 新增检测任务和实例分割任务的预测结果可视化、以及预测错误原因分析，辅助分析模型效果
  - 新增检测任务的负样本优化，抑制背景区域的误检
  - 完善语义分割任务的预测结果，支持返回预测类别和归一化后的预测置信度
  - 完善图像分类任务的预测结果，支持返回归一化后的预测置信度
* 预测部署
  - 完备PaddleX python预测部署, PaddleX模型使用2个API即可快速完成部署
  - PaddleX C++部署全面升级，支持飞桨视觉套件PaddleDetection、PaddleClas、PaddleSeg、PaddleX的端到端统一部署能力
  - 全新发布Manufacture SDK，提供工业级多端多平台部署加速的预编译飞桨部署开发包（SDK），通过配置业务逻辑流程文件即可以低代码方式快速完成推理部署
* PaddleX GUI
  - 升级PaddleX GUI，支持30系列显卡
  - 目标检测任务新增模型PP-YOLO V2, COCO test数据集精度达到49.5%、V100预测速度达到68.9 FPS
  - 目标检测任务新增4.2MB的超轻量级模型PP-YOLO tiny
  - 语义分割任务新增实时分割模型BiSeNetV2
  - 新增导出API训练脚本功能，无缝切换PaddleX API训练
* 产业实践案例
  - 新增以目标检测任务为主的钢筋计数、缺陷检测案例教程
  - 新增以实例分割任务为主的机械手抓取案例教程
  - 新增串联目标检测、语义分割、传统视觉算法的工业表计读数的训练和部署案例教程
  - 新增Windows系统下使用C#语言部署案例教程

### PaddleX v2.0.0rc0(5.19/2021)
* 全面支持飞桨2.0动态图，更易用的开发模式
* 目标检测任务新增[PP-YOLOv2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolov2.py), COCO test数据集精度达到49.5%、V100预测速度达到68.9 FPS 
* 目标检测任务新增4.2MB的超轻量级模型[PP-YOLO tiny](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/object_detection/ppyolotiny.py)
* 语义分割任务新增实时分割模型[BiSeNetV2](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/tutorials/train/semantic_segmentation/bisenetv2.py)
* C++部署模块全面升级
    * PaddleInference部署适配2.0预测库[（使用文档）](https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/deploy/cpp)
    * 支持飞桨[PaddleDetection]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddledetection.md)、[PaddleSeg]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleseg.md)、[PaddleClas](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleclas.md)以及PaddleX的模型部署
    * 新增基于PaddleInference的GPU多卡预测[（使用文档）](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/demo/multi_gpu_model_infer.md)
    * GPU部署新增基于ONNX的的TensorRT高性能加速引擎部署方式
    * GPU部署新增基于ONNX的Triton服务化部署方式[（使用文档）](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/compile/triton/docker.md)

### PaddleX v1.3.0(12.19/2020)

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

### PaddleX v1.2.0(9.9/2020)
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

### PaddleX v1.1.0(7.13/2020)
- 模型更新
> - 新增语义分割模型HRNet、FastSCNN
> - 目标检测FasterRCNN、实例分割MaskRCNN新增backbone HRNet
> - 目标检测/实例分割模型新增COCO数据集预训练模型
> - 集成X2Paddle，PaddleX所有分类模型和语义分割模型支持导出为ONNX协议
- 模型部署更新
> - 模型加密增加支持Windows平台
> - 新增Jetson、PaddleLite模型部署预测方案
> - C++部署代码新增batch批预测，并采用OpenMP对预处理进行并行加速
- 新增2个PaddleX产业案例
> - 人像分割案例
> - 工业表计读数案例
- 新增数据格式转换功能，LabelMe、精灵标注助手和EasyData平台标注的数据转为PaddleX支持加载的数据格式
- PaddleX文档更新，优化文档结构


### PaddleX v1.0.0(5.21/2020)

- **全流程打通**
  - **数据准备**：支持[EasyData智能数据服务平台](https://ai.baidu.com/easydata/)数据协议，通过平台便捷完成智能标注,低质数据清洗工作, 同时兼容主流标注工具协议, 助力开发者更快完成数据准备工作。
  - **模型训练**：集成[PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)视觉开发套件，丰富的高质量预训练模型，更快实现工业级模型效果。
  - **模型调优**：内置模型可解释性模块、[VisualDL](https://github.com/PaddlePaddle/VisualDL)可视化分析组件, 提供丰富的信息更好地理解模型，优化模型。
  - **多端安全部署**：内置[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)模型压缩工具和**模型加密部署模块**，结合Paddle Inference或[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)便捷完成高性能的多端安全部署。

- **融合产业实践**
  - 精选飞桨产业实践的成熟模型结构，开放案例实践教程，加速开发者产业落地。

- **易用易集成**
  - 统一易用的全流程API，5步完成模型训练，10行代码实现Python/C++高性能部署。
  - 提供以PaddleX为核心集成的跨平台可视化工具PaddleX-GUI，快速体验飞桨深度学习全流程。

