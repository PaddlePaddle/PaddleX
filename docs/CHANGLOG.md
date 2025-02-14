---
comments: true
---

# 版本更新信息

## 最新版本信息
### PaddleX v3.0.0rc0(2.14/2025)
PaddleX 3.0 rc0 全面适配 PaddlePaddle 3.0rc0 版本，新增10+条产线，40+个模型，优化模型和产线API，多硬件适配更多模型。全面升级高性能推理和服务化部署能力。具体新增能力如下：

- 新增产线：
  - 新增[文档预处理产线](pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.md)，支持将矫正旋转和扭曲的文档图像。
  - 新增[文档场景信息抽取v4产线](pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.md)，在文档场景信息抽取v3产线的基础上，融合了多模态能力，增强了OCR识别能力，优化了Prompt，最终文档信息抽取的准确率提升15个百分点。支持本地大模型OpenAI接口调用。
  - 新增[通用版面解析v2产线](pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.md)，PP-StructureV3 的核心方案。在通用版面解析v1产线的基础上，优化了版面区域检测、表格识别、公式识别、阅读顺序恢复的能力，支持将不同类型的文档图像和文档PDF文件转换为标准的Markdown文件，在大多数场景的文档恢复能力表现强劲。
  - 新增[通用表格识别v2产线](pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.md)，采用了“表格分类+表格结构识别+单元格检测”的多模型串联组网方案，实现更高精度的端到端表格识别。
  - 新增[旋转框检测产线](pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.md)，支持对旋转目标进行检测。
  - 新增[人体关键点检测产线](pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.md)，支持精确获取人体的关键点位置，如肩膀、肘部、膝盖等，从而进行姿态估计和行为识别。
  - 新增[开放词汇目标检测产线](pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.md)，支持对开放域目标进行检测，并预测类别。
  - 新增[开放词汇分割产线](pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.md)，支持对开放域目标进行图像分割。
  - 新增[通用视频检测产线](pipeline_usage/tutorials/video_pipelines/video_detection.md)，支持高效提取视频中的空间和时间特征，实现视频中目标的精准识别和定位。
  - 新增[通用视频分类产线](pipeline_usage/tutorials/video_pipelines/video_classification.md)，支持提取视频中的时空特征并进行准确分类。
  - 新增[多语种语音识别产线](pipeline_usage/tutorials/speech_pipelines/multilingual_speech_recognition.md)，支持将人类口述的多种语言自动转换为相应的文本或命令。
  - 新增[3D多模态融合检测产线](pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.md)，支持输入多种传感器（激光雷达、环视RGB相机等）数据，通过深度学习等方法对数据进行处理，输出三维空间中物体的位置、形状、朝向、类别等信息。

- 新增模型：
  - 新增 OCR 类模型 28 个，其中包含兼顾高精度和高效率的自研版面区域检测模型 PP-DocLayout 系列、兼顾高精度和高效率的自研公式识别模型 PP-FormulaNet 系列、自研表格结构识别模型 SLANeXt 系列、更高识别精度的自研文本识别模型 PP-OCRv4_server_rec_doc 模型等。
  - 新增 CV 类模型 11 个，新增了 3D 多模态融合检测模型、开放词汇目标检测和分割模型、旋转框检测模型、人体关键点检测模型等。
  - 新增 Speech 类模型 5 个，新增了 Whisper 系列的 5 个模型。
  - 新增 Video 类模型 4 个，包含视频检测模型 1 个、视频分类模型 3 个。

- 模型和产线能力升级：
  - 模型和产线支持更多参数，如目标检测模型的类别阈值、文本检测模型的膨胀系数等，CV 类和 OCR 类模型支持 PDF 格式文件输入。
  - OCR 类产线支持文档前处理操作，如文档方向分类、文档矫正等，内置文本行方向分类模型。
  - 文档场景信息抽取 v3 产线支持标准 OpenAI 接口调用大语言模型，支持更多大语言模型的调用。
  - 优化使用体验，部分模型和产线接口发生变化，详情参考 [API升级文档](API_change_log/v3.0.0rc.md)。

- 多硬件支持：
  - 新增燧原 GCU 硬件的模型训推能力，支持模型数量90+，[GCU 模型列表](support_list/model_list_gcu.md)
  - 昇腾 NPU 新增模型数量 50+，[NPU 模型列表](support_list/model_list_npu.md)
  - 昆仑芯 XPU 新增模型数量 10+，[XPU 模型列表](support_list/model_list_xpu.md)
  - 寒武纪 MLU 新增模型数量 10+，[MLU 模型列表](support_list/model_list_mlu.md)
  - 海光 DCU 新增模型数量 30+，[DCU 模型列表](support_list/model_list_dcu.md)

- 多环境适配
  - 适配 Windows 系统，支持在 Windows 下使用 PaddleX 进行模型训练和推理。修复部分 Windows 系统下安装失败的问题。
  - 训练和推理全面适配 Python3.11，Python3.12。

- 部署能力全面升级：
  - 高性能推理：
    - 安装使用简化：支持使用 PaddleX CLI 一键安装高性能推理插件；使用高性能推理插件无需进行鉴权。
    - 跨平台支持：新增对 Windows 系统的支持。
    - 模型支持扩展：扩增支持模型数量，目前总计支持 220+ 个模型。
    - 核心代码开源：开源核心推理库 ultra-infer，便于开发者进行二次开发和定制。
  - 服务化部署：
    - 基础服务化部署方案升级：升级基础服务化部署方案，支持新增产线，并适配原有产线新增功能。
    - 高稳定性服务化部署方案支持：新增高稳定性服务化部署方案，支持灵活调整服务配置以优化服务性能，多种部署方案满足不同用户需求。

### PaddleX v3.0.0beta2(11.15/2024)
PaddleX 3.0 Beta2 全面适配 PaddlePaddle 3.0b2 版本。**新增通用图像识别、人脸识别、车辆属性识别和行人属性识别产线，同时新增 42 个模型开发全流程适配昇腾 910B，并全面支持[GitHub 站点文档](https://paddlepaddle.github.io/PaddleX/latest/index.html)。** 具体新增能力如下：

- 新增产线：
  - 新增[通用图像识别产线](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.md)，提供更强的特征提取模型，支持用户自定义图像数据库识别未知类别，相比当前开放域目标检测，可以自定义识别的总类更多。支持高性能推理和服务化部署；
  - 新增[人脸识别产线](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/face_recognition.md)，支持对人脸数据库的增加和删除，支持高性能推理和服务化部署；
  - 新增[车辆属性识别产线](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.md)，支持对图像中的车辆进行检测和属性的识别，当前支持的属性有颜色和车型。支持高性能推理和服务化部署；
  - 新增[行人属性识别线](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.md)，支持对图像中的行人进行检测和属性的识别，当前支持的属性有年龄、性别、穿着等。支持高性能推理和服务化部署。

- 新增能力：
  - 支持[GitHub 站点文档](https://paddlepaddle.github.io/PaddleX/latest/index.html)，支持用户搜索相关内容和对文档内容的评论；
  - 支持打印模型的推理benchmark信息，相关[文档](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/module_usage/instructions/benchmark.md)；
  - 新增 42 个模型开发全流程适配昇腾 910B，[模型列表](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta2/docs/support_list/model_list_npu.md)。

- 优化点：
  - 公式识别产线，支持 PDF 格式输入，支持公式识别结果的可视化；
  - 印章文本识别产线，支持 PDF 格式的输入；
  - 通用版面解析产线，优化保存图片的名称；
  - 预训练统一管理，将当前支持的模型的预训练统一管理，并内置到默认配置文件中；
  - 升级模型保存的格式，为高性能推理提供保障；
  - 优化部分模型的默认参数，为训练更高精度的模型提供保障。

- BugFix：
  - 修复部分文档表述错误或者不当的内容，修复部分 URL 失效的问题；
  - 修复文档方向分类推理模型的bug；
  - 修复部分高性能推理和服务化部署的bug；
  - 修复 SLANet、SLANet_plus 训练精度为 0 的bug。


### PaddleX v3.0.0beta1(9.30/2024)
PaddleX 3.0 Beta1 提供 200+ 模型通过极简的 Python API 一键调用；实现基于统一命令的模型全流程开发，并开源 PP-ChatOCRv3-doc 特色模型产线基础能力；支持 100+ 模型高性能推理和服务化部署，7 类重点视觉模型端侧部署；70+ 模型开发全流程适配昇腾 910B，15+ 模型开发全流程适配昆仑芯和寒武纪。

- <b>模型丰富一键调用：</b> 将覆盖文档图像智能分析、OCR、目标检测、时序预测等多个关键领域的 200+ 飞桨模型整合为 13 条模型产线，通过极简的 Python API 一键调用，快速体验模型效果。同时支持 20+ 单功能模块，方便开发者进行模型组合使用。
- <b>提高效率降低门槛：</b> 实现基于图形界面和统一命令的模型全流程开发，打造大小模型结合、大模型半监督学习和多模型融合的8条特色模型产线，大幅度降低迭代模型的成本。
- <b>多种场景灵活部署：</b> 支持高性能部署、服务化部署和端侧部署等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。
- <b>主流硬件高效支持：</b> 支持英伟达 GPU、昆仑芯、昇腾和寒武纪等多种主流硬件的无缝切换，确保高效运行。

### PaddleX v3.0.0beta(6.27/2024)
PaddleX 3.0beta 集成了飞桨生态的优势能力，覆盖 7 大场景任务，构建了 16 条模型产线，提供低代码开发模式，助力开发者在多种主流硬件上实现模型全流程开发。

- <b>基础模型产线（模型丰富，场景全面）：</b> 精选 68 个优质飞桨模型，涵盖图像分类、目标检测、图像分割、OCR、文本图像版面分析、时序预测等任务场景。
- <b>特色模型产线（显著提升效率）：</b> 提供大小模型结合、大模型半监督学习和多模型融合的高效解决方案。
- <b>低代码开发模式（便捷开发与部署）：</b> 提供零代码和低代码两种开发方式。
  - 零代码开发：用户通过图形界面（GUI）交互式提交后台训练任务，打通在线和离线部署，并支持以 API 形式调用在线服务。
  - 低代码开发：通过统一的 API 接口实现 16 条模型产线的全流程开发，同时支持用户自定义模型流程串联。
- <b>多硬件本地支持（兼容性强）：</b> 支持英伟达 GPU、昆仑芯、昇腾和寒武纪等多种硬件，纯离线使用。

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
    * 支持飞桨[PaddleDetection]( https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddledetection.md)、[PaddleSeg](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleseg.md)、[PaddleClas](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/deploy/cpp/docs/models/paddleclas.md)以及PaddleX的模型部署
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
 > - 新增工业质检产业案例，提供基于GPU和CPU两种部署场景下的工业质检方案，及与质检相关的优化策略

- <b>新增RestFUL API模块</b>
新增RestFUL API模块，开发者可通过此模块快速开发基于PaddleX的训练平台
 > - 增加基于RestFUL API的HTML Demo
 > - 增加基于RestFUL API的Remote版可视化客户端
新增模型通过OpenVINO的部署方案

### PaddleX v1.2.0(9.9/2020)
- 模型更新
  > - 新增目标检测模型PPYOLO
  > - FasterRCNN、MaskRCNN、YOLOv3、DeepLabv3p等模型新增内置COCO数据集预训练模型
  > - 目标检测模型FasterRCNN和MaskRCNN新增backbone HRNet_W18
  > - 语义分割模型DeepLabv3p新增backbone MobileNetV3_large_ssld

- 模型部署更新
  > - 新增模型通过OpenVINO的部署方案
  > - 新增模型在树莓派上的部署方案
  > - 优化PaddleLite Android部署的数据预处理和后处理代码性能
  > - 优化Paddle服务端C++代码部署代码，增加use_mkl等参数，通过mkldnn显著提升模型在CPU上的预测性能

- 产业案例更新
  > - 新增RGB图像遥感分割案例
  > - 新增多通道遥感分割案例
- 其它
  > - 新增数据集切分功能，支持通过命令行切分ImageNet、PascalVOC、MSCOCO和语义分割数据集
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

- <b>全流程打通</b>
  - <b>数据准备</b>：支持[EasyData智能数据服务平台](https://ai.baidu.com/easydata/)数据协议，通过平台便捷完成智能标注,低质数据清洗工作, 同时兼容主流标注工具协议, 助力开发者更快完成数据准备工作。
  - <b>模型训练</b>：集成[PaddleClas](https://github.com/PaddlePaddle/PaddleClas), [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)视觉开发套件，丰富的高质量预训练模型，更快实现工业级模型效果。
  - <b>模型调优</b>：内置模型可解释性模块、[VisualDL](https://github.com/PaddlePaddle/VisualDL)可视化分析组件, 提供丰富的信息更好地理解模型，优化模型。
  - <b>多端安全部署</b>：内置[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)模型压缩工具和<b>模型加密部署模块</b>，结合Paddle Inference或[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite)便捷完成高性能的多端安全部署。

- <b>融合产业实践</b>
  - 精选飞桨产业实践的成熟模型结构，开放案例实践教程，加速开发者产业落地。

- <b>易用易集成</b>
  - 统一易用的全流程API，5步完成模型训练，10行代码实现Python/C++高性能部署。
  - 提供以PaddleX为核心集成的跨平台可视化工具PaddleX-GUI，快速体验飞桨深度学习全流程。
