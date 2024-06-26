<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/63c6d059-234f-4a27-955e-ac89d81409ee" width="360" height ="55" alt="PaddleX" align="middle" />
</p>

<p align= "center"> PaddleX -- 飞桨低代码开发工具，以低代码的形式支持开发者快速实现产业实际项目落地 </p>

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20windows-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/hardware-intel cpu%2C%20gpu%2C%20xpu%2C%20npu%2C%20mlu-yellow.svg"></a>
</p>

## 简介
PaddleX3.0 是飞桨精选模型的低代码开发工具，支持国内外多款主流硬件的模型训练和推理，覆盖工业、能源、金融、交通、教育等全行业，助力开发者产业实践落地。

任务示例展示

## 📣 近期更新
🔥 PaddleX3.0 升级中，6 月正式发布，敬请期待，云端使用请前往飞桨 AI Studio 星河社区：https://aistudio.baidu.com/pipeline/mine ，点击「创建产线」开启使用。

## 🌟 特性

PaddleX 3.0 集成飞桨生态优势能力，覆盖7大场景任务，构建 16 条模型产线，提供低代码开发模式，助力开发者在不同主流硬件上进行模型全流程开发。

  - **基础模型产线（模型数量多，场景全）：** 精选 72 个飞桨优质模型，覆盖图像分类、目标检测、图像分割、OCR、文本图像版面分析、时序预测等场景任务
  - **特色模型产线（提效显著）：** 提供大小模型结合，大模型半监督学习和多模型融合显著提效方案
  - **低代码开发模式（便捷开发部署）：** 提供零代码和低代码两种开发方式。
     - 零代码开发通过用户图形界面（GUI）交互式提交后台训练任务，打通在线&离线部署，支持以 API 的形式调用在线服务。
     - 低代码开发，一套 API 接口实现 16 条模型产线全流程开发，同时支持用户自定义模型串联流程。
  - **本地端多硬件支持（兼容性强）：** 支持英伟达 GPU、昆仑芯、昇腾和寒武纪多硬件上，纯离线使用 

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/61c4738f-735e-4ceb-aa5f-1038d4506d1c">
</div>

## ⚡ 安装与快速开始
- [安装](./docs/tutorials/INSTALL.md)
- 快速开始
  - [单模型开发工具](./docs/tutorials/inference/model_inference_tools.md)
  - [模型产线开发工具](./docs/tutorials/inference/pipeline_inference_tools.md)

## 🛠️ PaddleX3.0 覆盖的模型和模型产线
  - [单模型列表](./docs/tutorials/models/support_model_list.md)
  - [模型产线列表](./docs/tutorials/pipelines/support_pipeline_list.md)

## 📖 零代码开发教程
- [云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)：支持开发者使用零代码产线产出高质量模型和部署包
- [教程《零门槛开发产业级 AI 模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)：提供产业级模型开发经验，并且用12个实用的产业实践案例，手把手带你零门槛开发产业级AI模型

## 📖 低代码开发教程

### 一、单模型开发工具 🚀
本节介绍 PaddleX3.0 单模型的全流程开发流程，包括数据准备、模型训练/评估、模型推理的使用方法。PaddleX3.0 支持的模型可以参考 [PaddleX 模型库](./docs/tutorials/models/support_model_list.md)。

#### 1. 快速体验
- [快速体验](./docs/tutorials/models/model_inference_tools.md)

#### 2. 数据准备
- [数据准备流程](./docs/tutorials/data/README.md)
- [数据标注](./docs/tutorials/data/annotation/README.md)
- [数据校验](./docs/tutorials/data/dataset_check.md)

#### 3. 模型训练
- [模型训练/评估](./docs/tutorials/base/README.md)
- [模型优化](./docs/tutorials/base/model_optimize.md)

#### 4. 模型推理
- [模型推理](./docs/tutorials/base/README.md)

### 二、模型产线开发工具 🔥
本节将介绍 PaddleX3.0 模型产线的全流程开发流程，包括数据准备、模型训练/评估、模型推理的使用方法。PaddleX3.0 支持的模型产线可以参考 [PaddleX 模型产线列表](./docs/tutorials/pipelines/support_pipeline_list.md)

## 🌟 多硬件支持
本项目支持在多种硬件上进行模型的开发，除了 GPU 外，当前支持的硬件还有**昆仑芯**、**昇腾芯**、**寒武纪芯**。只需添加一个配置设备的参数，即可在对应硬件上使用上述工具。

- 昇腾芯支持的模型列表请参考 [PaddleX 昇腾芯模型列表](./docs/tutorials/models/support_npu_model_list.md)。
- 昆仑芯支持的模型列表请参考 [PaddleX 昆仑芯模型列表](./docs/tutorials/models/support_xpu_model_list.md)。
- 寒武纪芯支持的模型列表请参考 [PaddleX 寒武纪芯模型列表](./docs/tutorials/models/support_mlu_model_list.md)。


## 👀 贡献代码

我们非常欢迎您为 PaddleX 贡献代码或者提供使用建议。如果您可以修复某个 issue 或者增加一个新功能，欢迎给我们提交 Pull Requests。


