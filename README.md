<p align="center">
  <img src="./docs/gui/images/paddlex.png" width="360" height ="55" alt="PaddleX" align="middle" />
</p>
 <p align= "center"> PaddleX -- 飞桨全流程开发工具，以低代码的形式支持开发者快速实现产业实际项目落地 </p>

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/release/PaddlePaddle/PaddleX.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.6+-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/QQ_Group-1045148026-52B6EF?style=social&logo=tencent-qq&logoColor=000&logoWidth=20"></a>
</p>

## 近期动态
2021.09.10 PaddleX发布2.0.0正式版本，详情内容请参考[版本更新文档](./docs/CHANGELOG.md)。
- 全新发布Manufacture SDK，提供工业级多端多平台部署加速的预编译飞桨部署开发包（SDK），通过配置业务逻辑流程文件即可以低代码方式快速完成推理部署。[欢迎体验](./deploy/cpp/docs/manufacture_sdk)
- PaddleX部署全面升级，支持飞桨视觉套件PaddleDetection、PaddleClas、PaddleSeg、PaddleX的端到端统一部署能力。[欢迎体验](./deploy/cpp)
- 发布产业实践案例：钢筋计数、缺陷检测、机械手抓取、工业表计读数、Windows系统下使用C#语言部署。[欢迎体验](./examples)
- 升级PaddleX GUI，支持30系列显卡、新增模型PP-YOLO V2、PP-YOLO Tiny 、BiSeNetV2。[欢迎体验](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/install.md#2-padldex-gui%E5%BC%80%E5%8F%91%E6%A8%A1%E5%BC%8F%E5%AE%89%E8%A3%85)

## 产品介绍
:hugs: PaddleX 集成飞桨智能视觉领域**图像分类**、**目标检测**、**语义分割**、**实例分割**任务能力，将深度学习开发全流程从**数据准备**、**模型训练与优化**到**多端部署**端到端打通，并提供**统一任务API接口**及**图形化开发界面Demo**。开发者无需分别安装不同套件，以**低代码**的形式即可快速完成飞桨全流程开发。

:factory: **PaddleX** 经过**质检**、**安防**、**巡检**、**遥感**、**零售**、**医疗**等十多个行业实际应用场景验证，沉淀产业实际经验，**并提供丰富的案例实践教程**，全程助力开发者产业实践落地。

<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/132805400-8479384f-32d0-4feb-a7eb-ffad90382524.jpg" width="800"  />
</p>

## 安装与快速体验
PaddleX提供了图像化开发界面、本地API、Restful-API三种开发模式。用户可根据自己的需求选择任意一种开始体验
- [PaddleX API开发模式](./docs/quick_start_API.md)
- [PaddleX Restful API开发模式](./docs/quick_start_Resful_API.md)
- [PadldeX GUI开发模式](./docs/quick_start_GUI.md)
- [快速产业部署](#4-模型部署)

## 产业级应用示例

- [安防](./examples/security)
    - [车流量计数](./examples/security/vehicles_counting)  |  [人流量计数](./examples/security/pedestrian_flow_detection)  |  [安全帽检测](./examples/security/helmet_detection)  |  [火灾烟雾检测](./examples/security/fire_smoke_detection)
- [工业视觉](./examples/industrial_vision)
    - [铝板缺陷检测](./examples/industrial_vision/aluminum_defect_detection)  |  [齿轮缺陷检测](./examples/industrial_vision/gear_defect_detection)  |  [表计读数](./examples/industrial_vision/meter_reader)  |  [钢筋计数](./examples/industrial_vision/rebar_count)  |  [视觉辅助定位抓取](./examples/industrial_vision/robot_grab)
- [交通](./examples/traffic)
    - [车道线检测/车辆/行人检测](./examples/traffic/lane_vehicle_pedestrian_detection) 
- [遥感](./examples/remote_sensing)
    - [地块检测](./examples/remote_sensing/block_detection)  |  [变化检测](./examples/remote_sensing/variation_detection) 
- [互联网](./examples/Internet)
    - [快递信息智能提取](./examples/Internet/information_extraction)  |  [文本情感分析](./examples/Internet/text_sentiment_analysis)  |  [文字识别](./examples/Internet/character_recognition)  |  [文本重建](./examples/Internet/text_rebuild) 
- [特殊场景解决方案](./examples/special_scenario)
    - [先检测后分割场景串联解决方案]() 
    - [先检测后分类场景串联解决方案]()
    - [先分类后检测场景串联解决方案]()
    - [先分类后分类场景串联解决方案]()
    - [分割+检测场景并联解决方案]() 

## PaddleX 使用文档
本文档介绍了PaddleX从数据准备、模型训练到模型裁剪量化，及最终部署的全流程使用方法。
<p align="center">
  <img src="https://user-images.githubusercontent.com/53808988/133368962-8b330ea8-2590-45db-af0b-a3eab57e3e4b.png" width="800"  />
</p>

### 1. 数据准备

- [数据格式说明](./docs/data/format/README.md)
- [数据标注](./docs/data/annotation/README.md)
- [数据格式转换](./docs/data/convert.md)
- [数据划分](./docs/data/split.md)

### 2. 模型训练/评估/预测

- API开发模式
  - [API文档](./docs/apis)
    - [数据集读取API](./docs/apis/datasets.md)
    - [数据预处理和数据增强API](./docs/apis/transforms/transforms.md)
    - [模型API/模型加载API](./docs/apis/models/README.md)
    - [预测结果可视化API](./docs/apis/visualize.md)
  - [模型训练与参数调整](tutorials/train)
    - [模型训练](tutorials/train)
    - [训练参数调整](./docs/parameters.md)
  - [VisualDL可视化训练指标](./docs/visualdl.md)
  - [加载训好的模型完成预测及预测结果可视化](./docs/apis/prediction.md)
- [Restful API开发模式](./docs/Resful_API/docs)
  - [使用说明](./docs/Resful_API/docs)
  - [如何基于Restful API搭建一套AI开发平台]()
- [GUI开发模式](./docs/quick_start_GUI.md)
  - 视频教程：[图像分类](./docs/quick_start_GUI.md/#视频教程) | [目标检测](./docs/quick_start_GUI.md/#视频教程) | [语义分割](./docs/quick_start_GUI.md/#视频教程) | [实例分割](./docs/quick_start_GUI.md/#视频教程)
  
### 3. 模型压缩

- [支持的模型列表]()
- [模型剪裁](tutorials/slim/prune)
- [模型量化](tutorials/slim/quantize)

### 4. 模型部署

- [部署模型导出](./docs/apis/export_model.md)
- [部署方式概览](./deploy)
  - [硬件使用说明]()
  - [本地部署]()
    - [OpenVINO部署]()
    - [C++部署](./deploy/cpp)
      - [Manufacture SDK](./deploy/cpp/docs/manufacture_sdk) : [WinC#-Demo]()  |  [LinuxQT-Demo]()
      - [Deployment SDK](./deploy/cpp/docs/deployment.md) : [WinC#-Demo](./examples/C%23_deploy)  |  [LinuxQT-Demo]()
    - [Python部署](./docs/python_deploy.md)
  - [边缘侧部署]()
    - [JetsonQT-Demo]()
    - [Android/iOS部署]()
  - [服务化部署]()
    - [Triton部署]()
- [模型加密](./deploy/cpp/docs/demo/decrypt_infer.md)
- [ONNX格式转换](./deploy/cpp/docs/compile)

### 5. 附录

- [PaddleX模型库](./docs/appendix/model_zoo.md)
- [PaddleX指标及日志](./docs/appendix/metrics.md)
- [无联网模型训练](./docs/how_to_offline_run.md)

## 常见问题汇总
- [GUI相关问题](./docs/FAQ/FAQ.md/#GUI相关问题)
- [API训练相关问题](./docs/FAQ/FAQ.md/#API训练相关问题)
- [推理部署问题](./docs/FAQ/FAQ.md/#推理部署问题)

## 交流与反馈

- 项目官网：https://www.paddlepaddle.org.cn/paddle/paddlex
- PaddleX用户交流群：957286141 (手机QQ扫描如下二维码快速加入)  
  <p align="center">
    <img src="./docs/gui/images/QR2.jpg" width="250" height ="360" alt="QR" align="middle" />
  </p>

## :hugs: 贡献代码:hugs:

我们非常欢迎您为PaddleX贡献代码或者提供使用建议。如果您可以修复某个issue或者增加一个新功能，欢迎给我们提交Pull Requests。

### 开发者贡献项目

* [工业相机实时目标检测GUI](https://github.com/xmy0916/SoftwareofIndustrialCameraUsePaddle)
（windows系统，基于pyqt5开发）
* [工业相机实时目标检测GUI](https://github.com/LiKangyuLKY/PaddleXCsharp)
（windows系统，基于C#开发）
