<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/63c6d059-234f-4a27-955e-ac89d81409ee" width="360" height ="55" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20windows-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/hardware-intel cpu%2C%20gpu%2C%20xpu%2C%20npu%2C%20mlu-yellow.svg"></a>
</p>

<h4 align="center">
  <a href=#-特性>🌟 特性</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐 在线体验</a>｜<a href=#️-快速开始>🚀 快速开始</a> | <a href=#-文档> 📖 教程</a> | <a href=#-模型产线列表> 🔥模型产线列表</a>
</h4>

## 🔍 简介


PaddleX 3.0是基于飞桨框架构建的一套AI模型低代码开发工具，它集成了众多**开箱即用的预训练模型**，可以实现模型从训练到推理的**全流程开发**，支持国内外**多款主流硬件**，助力AI 开发者进行产业实践。  

|                **通用图像分类**                 |                **通用目标检测**                 |                **通用语义分割**                 |                **通用实例分割**                 |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px">|
|                  **通用OCR**                   |                **通用表格识别**                 |               **通用场景信息抽取**               |               **文档场景信息抽取**               |
|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/04218629-4a7b-48ea-b815-977a05fbbb13" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px">|
|                  **时序预测**                   |                **时序异常检测**                 |                 **时序分类**                   |              **多模型融合时序预测**              |
|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0959d099-a17c-40bc-9c2b-13f4f5e24ddc" height="126px" width="180px">|

## 🌟 特性
  🎨 **模型丰富一键调用**：将覆盖文本图像智能分析、OCR、目标检测、时序预测等多个关键领域的**200+飞桨模型**整合为**14条模型产线**，通过极简的Python API一键调用，快速体验模型效果。同时支持**20+单功能模块**，方便开发者进行模型组合使用。

  🚀 **提高效率降低门槛**：实现基于图形界面和统一命令的模型**全流程开发**，打造大小模型结合、大模型半监督学习和多模型融合的**8条特色模型产线**，大幅度降低迭代模型的成本。  

  🌐 **多种场景灵活部署**：支持**高性能部署**、**服务化部署**和**端侧部署**等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。

  🔧 **主流硬件高效支持**：支持英伟达 GPU、昆仑芯、昇腾和寒武纪等**多种主流硬件**的无缝切换，确保高效运行。

## 📣 近期更新

* 🔥 **2024.9.30**，PaddleX 3.0 Beta1 开源版正式发布，提供**200+模型**通过极简的Python API一键调用；实现基于统一命令的**模型全流程开发**，并开源**PP-ChatOCRv3**特色模型产线基础能力；支持**100+模型高性能推理和服务化部署**（持续迭代中），7类重点视觉模型**端侧部署**；70+模型开发全流程适配昇腾910B，15+模型开发全流程适配昆仑芯和寒武纪
* 🔥 **2024.6.27**，PaddleX 3.0 Beta 开源版正式发布，支持以低代码的方式在本地端使用多种主流硬件进行产线和模型开发。
* 🔥 **2024.3.25**，PaddleX 3.0 云端发布，支持在AI Studio 星河社区 以零代码的方式【创建产线】使用。


 ## 📊 能力支持

|       | [在线体验](https://aistudio.baidu.com/pipeline/mine) | 快速推理 | 二次开发 | 高性能部署 | 服务化部署 | 端侧部署 |
|-------|---|---|---|---|---|---|
| [OCR](/docs_new/pipelines_tutorials/OCR.md)                                                | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像分类](/docs_new/pipelines_tutorials/image_classification.md)                          | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [目标检测](/docs_new/pipelines_tutorials/object_detection.md)                              | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [语义分割](/docs_new/pipelines_tutorials/semantic_segmentation.md)                         | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [实例分割](/docs_new/pipelines_tutorials/instance_segmentation.md)                         | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [表格识别](/docs_new/pipelines_tutorials/table_recognition.md)                             | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序预测](/docs_new/pipelines_tutorials/time_series_forecasting.md)                       | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序异常检测](/docs_new/pipelines_tutorials/time_series_anomaly_detection.md)             | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序分类](/docs_new/pipelines_tutorials/time_series_classification.md)                    | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像多标签分类](/docs_new/pipelines_tutorials/image_multi_label_lassification.md)         | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [小目标检测](/docs_new/pipelines_tutorials/small_object_detection.md)                      | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像识别](/docs_new/pipelines_tutorials/image_recognition.md)                             | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像异常检测](/docs_new/pipelines_tutorials/image_anomaly_detection.md)                   | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |
| [文档场景信息抽取](/docs_new/pipelines_tutorials/document_scene_information_extraction.md) | ✅                  | ✅        | ✅          | ✅          | ✅        | ✅ |






**注：上述能力是基于GPU/CPU的能力，其他硬件支持的能力如下：**
<details>
  <summary>👉 昇腾芯能力支持</summary>

||快速推理 | 二次开发 | 高性能部署 | 服务化部署 | 端侧部署 |
|--------------------------------------------------------------------------------|----------|----------|------------|------------|------------|
| [OCR](/docs_new/pipelines_tutorials/OCR.md)                                    | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像分类](/docs_new/pipelines_tutorials/image_classification.md)              | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [目标检测](/docs_new/pipelines_tutorials/object_detection.md)                  | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [语义分割](/docs_new/pipelines_tutorials/semantic_segmentation.md)             | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [实例分割](/docs_new/pipelines_tutorials/instance_segmentation.md)             | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [表格识别](/docs_new/pipelines_tutorials/table_recognition.md)                 | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序预测](/docs_new/pipelines_tutorials/time_series_forecasting.md)           | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序异常检测](/docs_new/pipelines_tutorials/time_series_anomaly_detection.md) | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序分类](/docs_new/pipelines_tutorials/time_series_classification.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像多标签分类](/docs_new/pipelines_tutorials/image_multi_label_lassification.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [小目标检测](/docs_new/pipelines_tutorials/small_object_detection.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像识别](/docs_new/pipelines_tutorials/image_recognition.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像异常检测](/docs_new/pipelines_tutorials/image_anomaly_detection.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [文档场景信息抽取](/docs_new/pipelines_tutorials/document_scene_information_extraction.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
</details>


<details>
  <summary>👉 昆仑芯能力支持</summary>

||快速推理 | 二次开发 | 高性能部署 | 服务化部署 | 端侧部署 |
|--------------------------------------------------------------------------------|----------|----------|------------|------------|------------|
| [OCR](/docs_new/pipelines_tutorials/OCR.md)                                    | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像分类](/docs_new/pipelines_tutorials/image_classification.md)              | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [目标检测](/docs_new/pipelines_tutorials/object_detection.md)                  | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [语义分割](/docs_new/pipelines_tutorials/semantic_segmentation.md)             | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [实例分割](/docs_new/pipelines_tutorials/instance_segmentation.md)             | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [表格识别](/docs_new/pipelines_tutorials/table_recognition.md)                 | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序预测](/docs_new/pipelines_tutorials/time_series_forecasting.md)           | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序异常检测](/docs_new/pipelines_tutorials/time_series_anomaly_detection.md) | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序分类](/docs_new/pipelines_tutorials/time_series_classification.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像多标签分类](/docs_new/pipelines_tutorials/image_multi_label_lassification.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [小目标检测](/docs_new/pipelines_tutorials/small_object_detection.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像识别](/docs_new/pipelines_tutorials/image_recognition.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像异常检测](/docs_new/pipelines_tutorials/image_anomaly_detection.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [文档场景信息抽取](/docs_new/pipelines_tutorials/document_scene_information_extraction.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
</details>

<details>
  <summary>👉 寒武纪能力支持</summary>

||快速推理 | 二次开发 | 高性能部署 | 服务化部署 | 端侧部署 |
|--------------------------------------------------------------------------------|----------|----------|------------|------------|------------|
| [OCR](/docs_new/pipelines_tutorials/OCR.md)                                    | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像分类](/docs_new/pipelines_tutorials/image_classification.md)              | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [目标检测](/docs_new/pipelines_tutorials/object_detection.md)                  | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [语义分割](/docs_new/pipelines_tutorials/semantic_segmentation.md)             | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [实例分割](/docs_new/pipelines_tutorials/instance_segmentation.md)             | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [表格识别](/docs_new/pipelines_tutorials/table_recognition.md)                 | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序预测](/docs_new/pipelines_tutorials/time_series_forecasting.md)           | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序异常检测](/docs_new/pipelines_tutorials/time_series_anomaly_detection.md) | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [时序分类](/docs_new/pipelines_tutorials/time_series_classification.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像多标签分类](/docs_new/pipelines_tutorials/image_multi_label_lassification.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [小目标检测](/docs_new/pipelines_tutorials/small_object_detection.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像识别](/docs_new/pipelines_tutorials/image_recognition.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [图像异常检测](/docs_new/pipelines_tutorials/image_anomaly_detection.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
| [文档场景信息抽取](/docs_new/pipelines_tutorials/document_scene_information_extraction.md)        | ✅        | ✅        | ✅          | ✅          | ✅        | ✅ |
</details>


* PaddleX的各个产线均支持**快速推理**，您可以快速体验各个产线的预训练效果，如果您对产线的预训练效果满意，可以直接对产线进行**集成部署**，如果不满意，您也可以对产线中的单功能模块进行**二次开发**提升产线效果。详细请参考[文档](#-文档)
* 此外，[PaddleX星河零代码产线](https://aistudio.baidu.com/pipeline/mine)为开发者提供的基于图形用户界面(GUI)的全流程高效模型训练与部署工具。开发者**无需代码开发经验**，只需要准备符合产线要求的数据集即可**快速启动模型训练**,详细可以参考[零代码产线教程](https://ai.baidu.com/ai-doc/AISTUDIO/6lu57ycbb)


## ⏭️ 快速开始

### 🛠️ 安装

> ❗安装PaddleX前请先确保您有基础的Python运行环境，如果您还未安装Python环境，可以参考[运行环境准备](/docs_new/installation/installation.md#1-运行环境准备)进行安装

```python
# 您的机器安装的是CUDA 11，请运行以下命令安装
pip install paddlepaddle-gpu
# 您的机器是CPU，请运行以下命令安装
pip install paddlepaddle
...
```
  
更多安装方式参考[PaddleX安装教程](/docs_new/installation/installation.md)

### 💻 命令行使用

一行命令即可快速体验产线效果，以通用OCR产线为例：
```ruby
paddlex --pipeline image_classification --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/garbage_demo.png --device gpu:0
```

只需指定三个参数：
* `pipeline`：产线名称
* `input`：待处理的输入图片的本地路径或URL
* `device`: 使用的GPU序号（例如`gpu:0`表示使用第0块GPU），也可选择使用CPU（`cpu`）



其他产线的命令行使用，只需将`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：

<details>
  <summary>👉 更多产线的命令行使用</summary>

| 产线名称     | 对应参数                 | 详细说明 |
|----------|----------------------|------|
| 通用图像分类产线 | `image_classification` |   [通用图像分类产线命令行使用说明](/docs_new/pipelines_tutorials/image_classification.md)   |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |

</details>

### 📝 Python脚本使用

几行代码即可完成产线的快速推理，以通用OCR产线为例：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ocr")
output = pipeline.predict("pre_image.jpg")
for batch in output:
    for item in batch:
        res = item['result']
        res.print()
        res.save_to_img("./output/")
        res.save_to_json("./output/")
```
执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 调用产线对象的`predict` 方法进行推理预测
* 对预测结果进行处理

其他产线的Python脚本使用，只需将`create_pipeline()`方法的`pipeline`参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：
<details>
  <summary>👉 更多产线的Python脚本使用</summary>

| 产线名称     | 对应参数                 | 详细说明 |
|----------|----------------------|------|
| 通用图像分类产线 | `image_classification` |   [通用图像分类产线Python脚本使用说明](/docs_new/pipelines_tutorials/image_classification.md)   |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |
|          |                      |      |

</details>

更多的产线开发步骤请参考[完整文档](#-文档)

## 📖 文档
<details>
  <summary> <b> 安装 </summary>

* [PaddleX安装教程](/docs_new/installation/installation.md)  

...
</details>
<details>
  <summary> <b> 产线使用教程 </summary>
</details>
<details>
  <summary> <b> 单功能模块开发教程 </summary>
</details>

## 🔥 模型产线列表
<details>
  <summary> <b>通用OCR产线 </summary>

| 任务模块 | 模型            | 精度  | GPU推理耗时（ms） | CPU推理耗时 | 模型存储大小（M) | 
|----------|---------------------|-------|-------------------|-------------|------------------|
| 文本检测 | PP-OCRv4_mobile_det | 77.79 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_det | 82.69 | 22.20346          | 2662.158    | 198              | 
| 文本识别 | PP-OCRv4_mobile_rec | 78.20 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_rec | 79.20 | 22.20346          | 2662.158    | 198              | 

**注：文本检测模型精度指标为 Hmean(%)，文本识别模型精度指标为 Accuracy(%)。**

</details>

<details>
  <summary> <b> 通用图像分类产线 </summary>

| 任务模块 | 模型            | 精度  | GPU推理耗时（ms） | CPU推理耗时 | 模型存储大小（M) | 
|----------|---------------------|-------|-------------------|-------------|------------------|
| 文本检测 | PP-OCRv4_mobile_det | 77.79 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_det | 82.69 | 22.20346          | 2662.158    | 198              | 
| 文本识别 | PP-OCRv4_mobile_rec | 78.20 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_rec | 79.20 | 22.20346          | 2662.158    | 198              | 

**注：文本检测模型精度指标为 Hmean(%)，文本识别模型精度指标为 Accuracy(%)。**

</details>

<details>
  <summary> <b> 通用目标检测产线 </summary>
</details>

<details>
  <summary> <b> 通用实例分割产线 </summary>
</details>

<details>
  <summary> <b> 通用语义分割产线 </summary>
</details>

<details>
  <summary> <b> 通用表格识别产线 </summary>
</details>

<details>
  <summary> <b> 通用时序预测产线 </summary>
</details>

<details>
  <summary> <b> 通用时序异常检测产线 </summary>
</details>

<details>
  <summary> <b> 通用时序分类产线 </summary>
</details>

<details>
  <summary> <b> 通用图像多标签分类产线 </summary>
</details>

<details>
  <summary> <b> 小目标检测分类产线 </summary>
</details>

<details>
  <summary> <b> 通用图像识别产线 </summary>
</details>

<details>
  <summary> <b> 图像异常检测产线 </summary>
</details>

<details>
  <summary> <b> 文档场景信息抽取产线 </summary>
</details>

## 🤔 FAQ
## 💬 Discussion
## 📄 许可证书