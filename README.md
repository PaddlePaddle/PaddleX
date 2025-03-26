<p align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/logo.png" width="735" height ="200" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Python-3.8~3.12-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/OS-Linux%2C%20Windows%2C%20Mac-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/Hardware-CPU%2C%20GPU%2C%20XPU%2C%20NPU%2C%20MLU%2C%20DCU-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=#-特性>🌟 特性</a> | <a href=https://aistudio.baidu.com/application/center/app?tag=%E5%85%A8%E9%83%A8&flod=86503>🌐 在线体验</a>｜<a href=#️-快速开始>🚀 快速开始</a> | <a href=https://paddlepaddle.github.io/PaddleX/latest/index.html> 📖 文档</a> | <a href=#-能力支持> 🔥能力支持</a> | <a href=https://paddlepaddle.github.io/PaddleX/latest/support_list/models_list.html> 📋 模型列表</a>

</h4>

<h5 align="center">
  <a href="README.md">🇨🇳 简体中文</a> | <a href="README_en.md">🇬🇧 English</a></a>
</h5>

## 🔍 简介

PaddleX 3.0 是基于飞桨框架构建的低代码开发工具，它集成了众多**开箱即用的预训练模型**，可以实现模型从训练到推理的**全流程开发**，支持国内外**多款主流硬件**，助力AI 开发者进行产业实践。

|                                                            [**通用图像分类**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_classification.html)                                                            |                                                            [**图像多标签分类**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html)                                                            |                                                            [**通用目标检测**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/object_detection.html)                                                            |                                                            [**通用实例分割**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html)                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/multilabel_cls.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px"> |
|                                                              [**通用语义分割**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html)                                                               |                                                            [**图像异常检测**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html)                                                            |                                                         [ **通用OCR**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html)                                                          |                                                          [**通用表格识别**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html)                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/image_anomaly_detection.png" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px"> |
|                                                              [**文本图像智能分析**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.html)                                                              |                                                            [**时序预测**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html)                                                            |                                                              [**时序异常检测**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html)                                                              |                                                         [**时序分类**](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html)                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> |

## 🌟 特性
  🎨 **模型丰富一键调用**：将覆盖文本图像智能分析、OCR、目标检测、时序预测等多个关键领域的 **200+ 飞桨模型**整合为 **20 条模型产线**，通过极简的 Python API 一键调用，快速体验模型效果。同时支持 **20+ 单功能模块**，方便开发者进行模型组合使用。

  🚀 **提高效率降低门槛**：实现基于统一命令和图形界面的模型**全流程开发**，打造大小模型结合、大模型半监督学习和多模型融合的[**8 条特色模型产线**](https://aistudio.baidu.com/intro/paddlex)，大幅度降低迭代模型的成本。

  🌐 **多种场景灵活部署**：支持**高性能推理**、**服务化部署**和**端侧部署**等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。

  🔧 **主流硬件高效支持**：支持英伟达 GPU、昆仑芯、昇腾和寒武纪等**多种主流硬件**的无缝切换，确保高效运行。

## 📣 近期更新

🔥🔥 **2025.2.14，PaddleX v3.0.0rc0 重磅升级。** 本次版本全面适配 PaddlePaddle 3.0rc0及以上版本，核心升级如下：

- **新增 12 条高价值产线，重磅推出自研 [通用版面解析v3产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html)、[PP-ChatOCRv4-doc产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html)、[表格识别v2产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html)**。此外新增了文档处理、旋转框检测、开放词汇检测/分割、视频分析、多语种语音识别、3D 等场景的产线。

- **扩充 48 个前沿模型，包括重磅推出的 OCR 领域的版面区域检测模型 [PP-DocLayout](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html)、公式识别模型 [PP-FormulaNet](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/formula_recognition.html)，表格结构识别模型 [SLANeXt](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_structure_recognition.html)，文本识别模型 [PP-OCRv4_server_rec_doc](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html)**。CV 领域的 3D 检测、人体关键点、开放词汇检测/分割模型，以及语音识别领域的 Whisper 系列等模型。

- **优化和升级模型和产线的推理 API：** 支持更多参数的配置，提升模型和产线推理的灵活性，[详情](https://paddlepaddle.github.io/PaddleX/latest/API_change_log/v3.0.0rc.html)。

- **多硬件支持扩展：** 新增燧原 GCU 支持（90+模型），昇腾 NPU/昆仑芯 XPU/寒武纪 MLU/海光 DCU 模型数量显著提升。

- **全场景部署能力升级：**
  - **高性能推理支持一键安装、Windows 系统及 220+ 模型，核心库 ultra-infer 开源；**
  - **服务化部署新增高稳定性方案，支持动态配置优化。**

- **系统兼容性增强：** 适配 Windows 训练/推理，全面支持 Python 3.11/3.12。

🔥 **2024.11.15**，PaddleX 3.0 Beta2 开源版正式发布，全面适配 PaddlePaddle 3.0b2 版本。**新增通用图像识别、人脸识别、车辆属性识别和行人属性识别产线，同时新增 42 个模型开发全流程适配昇腾 910B，并全面支持[GitHub 站点文档](https://paddlepaddle.github.io/PaddleX/latest/index.html)。**

🔥 **2024.9.30**，PaddleX 3.0 Beta1 开源版正式发布，提供 **200+ 模型** 通过极简的 Python API 一键调用；实现基于统一命令的模型全流程开发，并开源 **PP-ChatOCRv3** 特色模型产线基础能力；支持 **100+ 模型高性能推理和服务化部署**（持续迭代中），**4条模型产线8个重点视觉模型端侧部署**；**100+ 模型开发全流程适配昇腾 910B**，**39+ 模型开发全流程适配昆仑芯和寒武纪**。


🔥 **2024.6.27**，PaddleX 3.0 Beta 开源版正式发布，支持以低代码的方式在本地端使用多种主流硬件进行产线和模型开发。

🔥 **2024.3.25**，PaddleX 3.0 云端发布，支持在 AI Studio 星河社区 以零代码的方式【创建产线】使用。


 ## 🔠 模型产线说明

 **PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。**


 ## 📊 能力支持


PaddleX的各个产线均支持本地**快速推理**，部分模型支持在[AI Studio星河社区](https://aistudio.baidu.com/overview)上进行**在线体验**，您可以快速体验各个产线的预训练模型效果，如果您对产线的预训练模型效果满意，可以直接对产线进行[高性能推理](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/high_performance_inference.html)/[服务化部署](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/serving.html)/[端侧部署](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/edge_deploy.html)，如果不满意，您也可以使用产线的**二次开发**能力，提升效果。完整的产线开发流程请参考[PaddleX产线使用概览](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/pipeline_develop_guide.html)或各产线使用[教程](#-文档)。


此外，PaddleX在[AI Studio星河社区](https://aistudio.baidu.com/overview)为开发者提供了基于[云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)的全流程开发工具, 点击【创建产线】，选择对应的任务场景和模型产线，就可以开启全流程开发。详细请参考[教程《零门槛开发产业级AI模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)

<table >
    <tr>
        <th>模型产线</th>
        <th>在线体验</th>
        <th>快速推理</th>
        <th>高性能推理</th>
        <th>服务化部署</th>
        <th>端侧部署</th>
        <th>二次开发</th>
        <th><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html">通用OCR</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html">文档场景信息抽取v3</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/182491/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html">通用表格识别</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/object_detection.html">通用目标检测</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html">通用实例分割</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_classification.html">通用图像分类</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html">通用语义分割</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html">时序预测</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html">时序异常检测</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html">时序分类</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/small_object_detection.html">小目标检测</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387975/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html">图像多标签分类</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387974/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html">公式识别</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387976/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html">印章文本识别</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387977/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.html">行人属性识别</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387978/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.html">车辆属性识别</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387979/webUI?source=appCenter">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html">图像异常检测</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.html">人体关键点检测</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.html">开放词汇检测</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.html">开放词汇分割</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.html">旋转目标检测</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.html">3D多模态融合检测</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html">通用表格识别v2</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html">通用版面解析</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html">通用版面解析v3</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html">文档图像预处理</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html">通用图像识别</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/face_recognition.html">人脸识别</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/speech_pipelines/multilingual_speech_recognition.html">多语种语音识别</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/video_pipelines/video_classification.html">通用视频分类</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/video_pipelines/video_detection.html">通用视频检测</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>


</table>

> ❗注：以上功能均基于 GPU/CPU 实现。PaddleX 还可在昆仑芯、昇腾、寒武纪和海光等主流硬件上进行快速推理和二次开发。下表详细列出了模型产线的支持情况，具体支持的模型列表请参阅[模型列表(昆仑芯XPU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_xpu.html)/[模型列表(昇腾NPU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_npu.html)/[模型列表(寒武纪MLU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_mlu.html)/[模型列表(海光DCU)](https://paddlepaddle.github.io/PaddleX/latest/support_list/model_list_dcu.html)。我们正在适配更多的模型，并在主流硬件上推动高性能和服务化部署的实施。

🔥🔥 **国产化硬件能力支持**

<table>
  <tr>
    <th>模型产线</th>
    <th>昇腾 910B</th>
    <th>昆仑芯 R200/R300</th>
    <th>寒武纪 MLU370X8</th>
    <th>海光 Z100/K100AI</th>
  </tr>
  <tr>
    <td>通用OCR</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>通用表格识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用目标检测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>通用实例分割</td>
    <td>✅</td>
    <td>🚧</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用图像分类</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>通用语义分割</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>时序预测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>时序异常检测</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>时序分类</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>图像多标签分类</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>行人属性识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>车辆属性识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>通用图像识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>印章文本识别</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>图像异常检测</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
  <tr>
    <td>人脸识别</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</table>

## ⏭️ 快速开始

### 🛠️ 安装

> ❗在安装 PaddleX 之前，请确保您已具备基本的 **Python 运行环境**（注：目前支持 Python 3.8 至 Python 3.12）。PaddleX 3.0-rc0 版本依赖的 PaddlePaddle 版本为 3.0.0rc0及以上版本，请在使用前务必保证版本的对应关系。

* **安装 PaddlePaddle**
```bash
# CPU 版本
python -m pip install paddlepaddle==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU 版本，需显卡驱动程序版本 ≥545.23.06（Linux）或 ≥545.84（Windows）
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗无需关注物理机上的 CUDA 版本，只需关注显卡驱动程序版本。更多飞桨 Wheel 版本信息，请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html)。

* **安装PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl
```

> ❗ 更多安装方式参考 [PaddleX 安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)

### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

PaddleX的每一条产线对应特定的参数，您可以在各自的产线文档中查看具体的参数说明。每条产线需指定必要的三个参数：
* `pipeline`：产线名称或产线配置文件
* `input`：待处理的输入文件（如图片）的本地路径、目录或 URL
* `device`：使用的硬件设备及序号（例如`gpu:0`表示使用第 0 块 GPU），也可选择使用 NPU(`npu:0`)、 XPU(`xpu:0`)、CPU(`cpu`)等。


以通用 OCR 产线为例：
```bash
paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
```
<details>
  <summary><b>👉 点击查看运行结果 </b></summary>

```bash
{'res': {'input_path': 'general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': False}, 'angle': 0},'dt_polys': [array([[ 3, 10],
       [82, 10],
       [82, 33],
       [ 3, 33]], dtype=int16), ...], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, ...], 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.99*', ...], 'rec_scores': [0.8980069160461426,  ...], 'rec_polys': [array([[ 3, 10],
       [82, 10],
       [82, 33],
       [ 3, 33]], dtype=int16), ...], 'rec_boxes': array([[  3,  10,  82,  33], ...], dtype=int16)}}
```

可视化结果如下：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)

</details>

其他产线的命令行使用，只需将 `pipeline` 参数调整为相应产线的名称，参数调整为对应的产线的参数即可。下面列出了每个产线对应的命令：

<details>
  <summary><b>👉 更多产线的命令行使用</b></summary>

| 产线名称           | 使用命令                                                                                                                                                                                    |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用图像分类       | `paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --device gpu:0 --save_path ./output/`                    |
| 通用目标检测       | `paddlex --pipeline object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_object_detection_002.png --threshold 0.5 --save_path ./output/ --device gpu:0`                            |
| 通用实例分割       | `paddlex --pipeline instance_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_instance_segmentation_004.png --threshold 0.5 --save_path ./output --device gpu:0`                  |
| 通用语义分割       | `paddlex --pipeline semantic_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/application/semantic_segmentation/makassaridn-road_demo.png --target_size -1 --save_path ./output --device gpu:0` |
| 图像多标签分类 | `paddlex --pipeline image_multilabel_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_image_classification_001.jpg --save_path ./output --device gpu:0`        |
| 小目标检测         | `paddlex --pipeline small_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/small_object_detection.jpg --threshold 0.5 --save_path ./output --device gpu:0`                            |
| 图像异常检测       | `paddlex --pipeline anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/uad_grid.png --save_path ./output --device gpu:0`                                              |
| 行人属性识别       | `paddlex --pipeline pedestrian_attribute_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pedestrian_attribute_002.jpg --save_path ./output/ --device gpu:0`                                              |
| 车辆属性识别       | `paddlex --pipeline vehicle_attribute_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_attribute_002.jpg --save_path ./output/ --device gpu:0`                                              |
| 3D多模态融合检测       | `paddlex --pipeline 3d_bev_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/det_3d/demo_det_3d/nuscenes_demo_infer.tar --device gpu:0 --save_path ./output/`                    |
| 人体关键点检测      | `paddlex --pipeline human_keypoint_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/keypoint_detection_001.jpg --det_threshold 0.5 --save_path ./output/ --device gpu:0`                    |
| 开放词汇检测       | `paddlex --pipeline open_vocabulary_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_detection.jpg --prompt "bus . walking man . rearview mirror ." --thresholds "{'text_threshold': 0.25, 'box_threshold': 0.3}" --save_path ./output --device gpu:0`                    |
| 开放词汇分割       | `paddlex --pipeline open_vocabulary_segmentation --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/open_vocabulary_segmentation.jpg --prompt_type box --prompt "[[112.9,118.4,513.8,382.1],[4.6,263.6,92.2,336.6],[592.4,260.9,607.2,294.2]]" --save_path ./output --device gpu:0`                    |
| 旋转目标检测       | `paddlex --pipeline rotated_object_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/rotated_object_detection_001.png --threshold 0.5 --save_path ./output --device gpu:0`                    |
| 通用OCR            | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                                      |
| 文档图像预处理            | `paddlex --pipeline doc_preprocessor --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg --use_doc_orientation_classify True --use_doc_unwarping True --save_path ./output --device gpu:0`                                                      |
| 通用表格识别       | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --save_path ./output --device gpu:0`                                      |
| 通用表格识别v2       | `paddlex --pipeline table_recognition_v2 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --save_path ./output --device gpu:0`                                      |
| 通用版面解析       | `paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                      |
| 通用版面解析v3       | `paddlex --pipeline PP-StructureV3 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                      |
| 公式识别       | `paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --use_layout_detection True --use_doc_orientation_classify False --use_doc_unwarping False --layout_threshold 0.5 --layout_nms True --layout_unclip_ratio  1.0 --layout_merge_bboxes_mode large --save_path ./output --device gpu:0`                                      |
| 印章文本识别       | `paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --use_doc_orientation_classify False --use_doc_unwarping False --device gpu:0 --save_path ./output`                                      |
| 时序预测       | `paddlex --pipeline ts_forecast --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_fc.csv --device gpu:0 --save_path ./output`                                                                   |
| 时序异常检测   | `paddlex --pipeline ts_anomaly_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_ad.csv --device gpu:0 --save_path ./output`                                                                    |
| 时序分类       | `paddlex --pipeline ts_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/ts/demo_ts/ts_cls.csv --device gpu:0 --save_path ./output`                                                                 |
| 多语种语音识别       | `paddlex --pipeline multilingual_speech_recognition --input https://paddlespeech.bj.bcebos.com/PaddleAudio/zh.wav --save_path ./output --device gpu:0`                                      |
| 通用视频分类       | `paddlex --pipeline video_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/general_video_classification_001.mp4 --topk 5 --save_path ./output --device gpu:0`                     |
| 通用视频检测       | `paddlex --pipeline video_detection --input https://paddle-model-ecology.bj.bcebos.com/paddlex/videos/demo_video/HorseRiding.avi --device gpu:0 --save_path ./output`                     |


</details>

### 📝 Python 脚本使用

几行代码即可完成产线的快速推理，统一的 Python 脚本格式如下：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[产线名称])
output = pipeline.predict([输入图片名称])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的 `predict()` 方法进行推理预测
* 对预测结果进行处理

其他产线的 Python 脚本使用，只需将 `create_pipeline()` 方法的 `pipeline` 参数调整为相应产线的名称，参数调整为对应的产线的参数即可。下面列出了每个产线对应的参数名称及详细的使用解释：
<details>
  <summary><b>👉 更多产线的Python脚本使用</b></summary>

| 产线名称           | 对应参数                           | 详细说明                                                                                                                                                         |
|--------------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 文档场景信息抽取v4   | `PP-ChatOCRv4-doc`                 | [文档场景信息抽取v4产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html#22-本地体验) |
| 文档场景信息抽取v3   | `PP-ChatOCRv3-doc`                 | [文档场景信息抽取v3产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.html#22-本地体验) |
| 通用图像分类       | `image_classification`             | [通用图像分类产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_classification.html#222-python脚本方式集成)                                |
| 通用目标检测       | `object_detection`                 | [通用目标检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/object_detection.html#222-python脚本方式集成)                                    |
| 通用实例分割       | `instance_segmentation`            | [通用实例分割产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html#222-python脚本方式集成)                               |
| 通用语义分割       | `semantic_segmentation`            | [通用语义分割产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html#222-python脚本方式集成)                               |
| 图像多标签分类 | `multi_label_image_classification` | [图像多标签分类产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html#22-python脚本方式集成)               |
| 小目标检测         | `small_object_detection`           | [小目标检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/small_object_detection.html#22-python脚本方式集成)                                 |
| 图像异常检测       | `anomaly_detection`                | [图像异常检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html#22-python脚本方式集成)                              |
| 通用图像识别       | `PP-ShiTuV2`                | [通用图像识别Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html#22-python脚本方式集成)                              |
| 人脸识别       | `face_recognition`                | [人脸识别Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/face_recognition.html#22-python脚本方式集成)                              |
| 车辆属性识别       | `vehicle_attribute_recognition`                | [车辆属性识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.html#22-python脚本方式集成)                              |
| 行人属性识别       | `pedestrian_attribute_recognition`                | [行人属性识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.html#22-python脚本方式集成)                              |
| 3D多模态融合检测       | `3d_bev_detection`             | [3D多模态融合检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.html#222-python脚本方式集成)                                |
| 人体关键点检测       | `human_keypoint_detection`             | [人体关键点检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.html#222-python脚本方式集成)                       |
| 开放词汇检测       | `open_vocabulary_detection`             | [开放词汇检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.html#212-python脚本方式集成)                                |
| 开放词汇分割       | `open_vocabulary_segmentation`             | [开放词汇分割产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.html#212-python脚本方式集成)                     |
| 旋转目标检测       | `rotated_object_detection`             | [旋转目标检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.html#212-python脚本方式集成)                                |
| 通用OCR            | `OCR`                              | [通用OCR产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html#222-python脚本方式集成)                                                     |
| 文档图像预处理            | `doc_preprocessor`                              | [文档图像预处理产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html#212-python脚本方式集成)                       |
| 通用表格识别       | `table_recognition`                | [通用表格识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html#22-python脚本方式集成)                                   |
| 通用表格识别v2      | `table_recognition_v2`                | [通用表格识别v2产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html#22-python脚本方式集成)                                   |
| 通用版面解析       | `layout_parsing`                | [通用版面解析产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html#22-python脚本方式集成)                                   |
| 通用版面解析v3      | `PP-StructureV3`                | [通用版面解析v3产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html#22-python脚本方式集成)                                   |
| 公式识别       | `formula_recognition`                | [公式识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html#22-python脚本方式集成)                                   |
| 印章文本识别       | `seal_recognition`                | [印章文本识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html#22-python脚本方式集成)                                   |
| 时序预测       | `ts_forecast`                            | [时序预测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html#222-python脚本方式集成)                    |
| 时序异常检测   | `ts_anomaly_detection`                            | [时序异常检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html#222-python脚本方式集成)          |
| 时序分类       | `ts_classification`                           | [时序分类产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html#222-python脚本方式集成)                 |
| 多语种语音识别       | `multilingual_speech_recognition`                           | [多语种语音识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/multilingual_speech_recognition.html#212-python脚本方式集成)                 |
| 通用视频分类       | `video_classification`                           | [通用视频分类产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/video_classification.html#22-python脚本方式集成)                 |
| 通用视频检测       | `video_detection`                           | [通用视频检测产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/video_detection.html#212-python脚本方式集成)                 |

</details>


## 📖 文档
<details>
  <summary> <b> ⬇️ 安装 </b></summary>

  * [📦 PaddlePaddle 安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/paddlepaddle_install.html)
  * [📦 PaddleX 安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)


</details>

<details open>
<summary> <b> 🔥 产线使用 </b></summary>

* [📑 PaddleX 产线使用概览](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/pipeline_develop_guide.html)

* <details open>
    <summary> <b> 📝 文本图像智能分析 </b></summary>

   * [📄 文档场景信息抽取v3产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html)
  </details>

* <details open>
    <summary> <b> 🔍 OCR </b></summary>

  * [📜 通用 OCR 产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html )
  * [📊 通用表格识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html )
  * [🗂️ 通用表格识别v2产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html)
  * [📰 通用版面解析产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html)
  * [🗞️ 通用版面解析产线v3使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html)
  * [📐 公式识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html)
  * [🖋️ 印章文本识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html)
  * [🖌️ 文档图像预处理产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html)
</details>

* <details open>
    <summary> <b> 🎥 计算机视觉 </b></summary>

   * [🖼️ 通用图像分类产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_classification.html)
   * [🎯 通用目标检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/object_detection.html)
   * [📋 通用实例分割产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/instance_segmentation.html)
   * [🗣️ 通用语义分割产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/semantic_segmentation.html)
   * [🏷️ 图像多标签分类产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_multi_label_classification.html)
   * [🔍 小目标检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/small_object_detection.html)
   * [🖼️ 图像异常检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/image_anomaly_detection.html)
   * [🔍 人体关键点检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/human_keypoint_detection.html)
   * [📚 开放词汇检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_detection.html)
   * [🎨 开放词汇分割产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/open_vocabulary_segmentation.html)
   * [🔄 旋转目标检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/rotated_object_detection.html)
   * [🌐 3D多模态融合检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/3d_bev_detection.html)
   * [🖼️ 通用图像识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/general_image_recognition.html)
   * [🆔人脸识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/face_recognition.html)
   * [🚗 车辆属性识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/vehicle_attribute_recognition.html)
   * [🚶‍♀️ 行人属性识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/cv_pipelines/pedestrian_attribute_recognition.html)


* <details open>
    <summary> <b> ⏱️ 时序分析</b> </summary>

   * [📈 时序预测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_forecasting.html)
   * [📉 时序异常检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.html)
   * [🕒 时序分类产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/time_series_pipelines/time_series_classification.html)
  </details>

* <details open>
    <summary> <b> 🎤 语音识别</b> </summary>

    * [🌐 多语种语音识别产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/speech_pipelines/multilingual_speech_recognition.html)

* <details open>
    <summary> <b> 🎥 视频识别</b> </summary>

    * [📈 通用视频分类产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/video_pipelines/video_classification.html)
    * [🔍 通用视频检测产线使用教程](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/video_pipelines/video_detection.html)

* <details>
    <summary> <b>🔧 相关说明文件</b> </summary>

   * [🖥️ PaddleX 产线命令行使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/instructions/pipeline_CLI_usage.html)
   * [📝 PaddleX 产线 Python 脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/instructions/pipeline_python_API.html)
  </details>

</details>

<details open>
<summary> <b> ⚙️ 单功能模块使用 </b></summary>

* <details open>
  <summary> <b> 🔍 OCR </b></summary>

  * [📝 文本检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_detection.html)
  * [🔖 印章文本检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/seal_text_detection.html)
  * [🔠 文本识别模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html)
  * [🗺️ 版面区域检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html)
  * [📊 表格结构识别模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_structure_recognition.html)
  * [📄 文档图像方向分类使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html)
  * [🔧 文本图像矫正模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_image_unwarping.html)
  * [📐 公式识别模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/formula_recognition.html)
  * [📊 表格单元格检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_cells_detection.html)
  * [📈 表格分类模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_classification.html)
  * [📝 文本行方向分类模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/textline_orientation_classification.html)

  </details>

* <details open>
  <summary> <b> 🖼️ 图像分类 </b></summary>

  * [📂 图像分类模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_classification.html)
  * [🏷️ 图像多标签分类模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_multilabel_classification.html)
  * [👤 行人属性识别模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.html)
  * [🚗 车辆属性识别模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/vehicle_attribute_recognition.html)

  </details>

* <details open>
  <summary> <b> 🏞️ 图像特征 </b></summary>

    * [🔗 图像特征模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/image_feature.html)
    * [😁 人脸特征模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/face_feature.html)
  </details>

* <details open>
  <summary> <b> 🎯 目标检测 </b></summary>

  * [🎯 目标检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/object_detection.html)
  * [📏 小目标检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/small_object_detection.html)
  * [🧑‍🤝‍🧑 人脸检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/face_detection.html)
  * [🔍 主体检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/mainbody_detection.html)
  * [🚶 行人检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/human_detection.html)
  * [🚗 车辆检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/vehicle_detection.html)
  * [🔄 旋转目标检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/rotated_object_detection.html)

* <details open>
  <summary> <b> 🌐 开放词汇目标检测 </b></summary>

  * [🌐 开放词汇目标检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/open_vocabulary_detection.html)
  </details>

* <details open>
  <summary> <b> 🎯 关键点检测 </b></summary>

  * [🚶‍♂️ 人体关键点检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/human_keypoint_detection.html)
  </details>


* <details open>
  <summary> <b> 🖼️ 图像分割 </b></summary>

  * [🗺️ 语义分割模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/semantic_segmentation.html)
  * [🔍 实例分割模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/instance_segmentation.html)
  * [🚨 图像异常检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/anomaly_detection.html)
  </details>

* <details open>
  <summary> <b> 🌐 开放词汇分割 </b></summary>

  * [🌐 开放词汇分割模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/open_vocabulary_segmentation.html)
  </details>

* <details open>
  <summary> <b> ⏱️ 时序分析 </b></summary>

  * [📈 时序预测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/time_series_modules/time_series_forecasting.html)
  * [🚨 时序异常检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/time_series_modules/time_series_anomaly_detection.html)
  * [🕒 时序分类模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/time_series_modules/time_series_classification.html)
  </details>

* <details open>
  <summary> <b> 📦 3D </b></summary>

  * [📦 3D多模态融合检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/cv_modules/3d_bev_detection.html)

* <details open>
  <summary> <b> 🎤 语音识别 </b></summary>

  * [🌐 多语种语音识别模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/speech_modules/multilingual_speech_recognition.html)

* <details open>
  <summary> <b> 🎥 视频识别 </b></summary>

  * [📈 视频分类模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/video_modules/video_classification.html)
  * [🔍 视频检测模块使用教程](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/video_modules/video_detection.html)

* <details>
  <summary> <b> 📄 相关说明文件 </b></summary>

  * [📝 PaddleX 单模型 Python 脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/module_usage/instructions/model_python_API.html)
  * [📝 PaddleX 通用模型配置文件参数说明](https://paddlepaddle.github.io/PaddleX/latest/module_usage/instructions/config_parameters_common.html)
  * [📝 PaddleX 时序任务模型配置文件参数说明](https://paddlepaddle.github.io/PaddleX/latest/module_usage/instructions/config_parameters_time_series.html)
  * [📝 PaddleX 3d任务模型配置文件参数说明](https://paddlepaddle.github.io/PaddleX/latest/module_usage/instructions/config_parameters_3d.html)
  </details>

</details>

<details open>
  <summary> <b> 🏗️ 模型产线部署 </b></summary>

  * [🚀 PaddleX 高性能推理指南](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/high_performance_inference.html)
  * [🖥️ PaddleX 服务化部署指南](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/serving.html)
  * [📱 PaddleX 端侧部署指南](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/edge_deploy.html)

</details>
<details open>
  <summary> <b> 🖥️ 多硬件使用 </b></summary>

  * [🔧 多硬件使用指南](https://paddlepaddle.github.io/PaddleX/latest/other_devices_support/multi_devices_use_guide.html)
  * [🖲️ 海光 DCU 飞桨安装教程](https://paddlepaddle.github.io/PaddleX/latest/other_devices_support/paddlepaddle_install_DCU.html)
  * [🔲 寒武纪 MLU 飞桨安装教程](https://paddlepaddle.github.io/PaddleX/latest/other_devices_support/paddlepaddle_install_MLU.html)
  * [💻 昇腾 NPU 飞桨安装教程](https://paddlepaddle.github.io/PaddleX/latest/other_devices_support/paddlepaddle_install_NPU.html)
  * [🔌 昆仑 XPU 飞桨安装教程](https://paddlepaddle.github.io/PaddleX/latest/other_devices_support/paddlepaddle_install_XPU.html)

</details>

<details>
  <summary> <b> 📝 产业实践教程&范例 </b></summary>

* [📑 文档场景信息抽取v3模型产线———论文文献信息抽取应用教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/document_scene_information_extraction%28layout_detection%29_tutorial.html)
* [📑 文档场景信息抽取v3模型产线———印章信息抽取应用教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/document_scene_information_extraction%28seal_recognition%29_tutorial.html)
* [📑 文档场景信息抽取v3模型产线———Deepseek应用教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/document_scene_information_extraction%28deepseek%29_tutorial.html)
* [📑 公式识别模型产线———后处理参数实践教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/formula_recognition_tutorial.html)
* [🖼️ 通用图像分类模型产线———垃圾分类教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/image_classification_garbage_tutorial.html)
* [🧩 通用实例分割模型产线———遥感图像实例分割教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/instance_segmentation_remote_sensing_tutorial.html)
* [👥 通用目标检测模型产线———行人跌倒检测教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/object_detection_fall_tutorial.html)
* [👗 通用目标检测模型产线———服装时尚元素检测教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/object_detection_fashion_pedia_tutorial.html)
* [🚗 通用 OCR 模型产线———车牌识别教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/ocr_det_license_tutorial.html)
* [✍️ 通用 OCR 模型产线———手写中文识别教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/ocr_rec_chinese_tutorial.html)
* [🗣️ 通用语义分割模型产线———车道线分割教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/semantic_segmentation_road_tutorial.html)
* [🛠️ 时序异常检测模型产线———设备异常检测应用教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/ts_anomaly_detection.html)
* [🎢 时序分类模型产线———心跳监测时序数据分类应用教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/ts_classification.html)
* [🔋 时序预测模型产线———用电量长期预测应用教程](https://paddlepaddle.github.io/PaddleX/latest/practical_tutorials/ts_forecast.html)

  </details>

## 🤔 FAQ

关于我们项目的一些常见问题解答，请参考[FAQ](https://paddlepaddle.github.io/PaddleX/latest/FAQ.html)。如果您的问题没有得到解答，请随时在 [Issues](https://github.com/PaddlePaddle/PaddleX/issues) 中提出
## 💬 Discussion

我们非常欢迎并鼓励社区成员在 [Discussions](https://github.com/PaddlePaddle/PaddleX/discussions) 板块中提出问题、分享想法和反馈。无论您是想要报告一个 bug、讨论一个功能请求、寻求帮助还是仅仅想要了解项目的最新动态，这里都是一个绝佳的平台。


## 📄 许可证书

本项目的发布受 [Apache 2.0 license](./LICENSE) 许可认证。
