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
  <a href=#-特性>🌟 特性</a> | <a href=https://aistudio.baidu.com/pipeline/mine>🌐 在线体验</a>｜<a href=#️-快速开始>🚀 快速开始</a> | <a href=#-文档> 📖 文档</a> | <a href=#-模型产线列表> 🔥模型产线列表</a>
</h4>

## 🔍 简介


PaddleX 3.0是基于飞桨框架构建的一套AI模型低代码开发工具，它集成了众多**开箱即用的预训练模型**，可以实现模型从训练到推理的**全流程开发**，支持国内外**多款主流硬件**，助力AI 开发者进行产业实践。  

|                                                            **通用图像分类**                                                            |                                                            **通用/小目标检测**                                                            |                                                            **通用语义分割**                                                            |                                                            **通用实例分割**                                                            |
|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------:|
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px"> |
|                                                              **通用多标签分类**                                                               |                                                            **通用OCR**                                                            |                                                          **通用表格识别**                                                          |                                                          **文档场景信息抽取v3**                                                          |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px"> |  <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px"> |
|                                                              **时序预测**                                                              |                                                            **时序异常检测**                                                            |                                                              **时序分类**                                                              |                                                         **图像异常检测**                                                         |
| <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> | <img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px"> |

## 🌟 特性
  🎨 **模型丰富一键调用**：将覆盖文本图像智能分析、OCR、目标检测、时序预测等多个关键领域的**200+飞桨模型**整合为**13条模型产线**，通过极简的Python API一键调用，快速体验模型效果。同时支持**20+单功能模块**，方便开发者进行模型组合使用。

  🚀 **提高效率降低门槛**：实现基于图形界面和统一命令的模型**全流程开发**，打造大小模型结合、大模型半监督学习和多模型融合的**8条特色模型产线**，大幅度降低迭代模型的成本。  

  🌐 **多种场景灵活部署**：支持**高性能部署**、**服务化部署**和**端侧部署**等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。

  🔧 **主流硬件高效支持**：支持英伟达 GPU、昆仑芯、昇腾和寒武纪等**多种主流硬件**的无缝切换，确保高效运行。

## 📣 近期更新

🔥 **2024.9.30**，PaddleX 3.0 Beta1 开源版正式发布，提供**200+模型**通过极简的Python API一键调用；实现基于统一命令的**模型全流程开发**，并开源**PP-ChatOCRv3**特色模型产线基础能力；支持**100+模型高性能推理和服务化部署**（持续迭代中），7类重点视觉模型**端侧部署**；70+模型开发全流程适配昇腾910B，15+模型开发全流程适配昆仑芯和寒武纪

🔥 **2024.6.27**，PaddleX 3.0 Beta 开源版正式发布，支持以低代码的方式在本地端使用多种主流硬件进行产线和模型开发。

🔥 **2024.3.25**，PaddleX 3.0 云端发布，支持在AI Studio 星河社区 以零代码的方式【创建产线】使用。


 ## 📊 能力支持

PaddleX的各个产线均支持在[AI Studio星河社区](https://aistudio.baidu.com/pipeline/mine)上**在线体验**和本地**快速推理**，您可以快速体验各个产线的预训练效果，如果您对产线的预训练效果满意，可以直接对产线进行**高性能部署/服务化部署/端侧部署**，如果不满意，您也可以对产线中的单功能模块进行**二次开发**提升产线效果。完整的产线开发流程请参考[PaddleX产线开发工具本地使用教程](/docs_new/pipelines_usage/pipeline_develop_guide.md)。

此外，为开发者提供的基于[云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)的全流程高效模型训练与部署工具。开发者**无需代码开发**，只需要准备符合产线要求的数据集即可**快速启动模型训练**,详细请参考[教程《零门槛开发产业级AI模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)


<table >
    <tr>
        <td></td>
        <td>在线体验</td>
        <td>快速推理</td>
        <td>高性能部署</td>
        <td>服务化部署</td>
        <td>端侧部署</td>
        <td>二次开发</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>目标检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>语义分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>实例分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>表格识别</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序预测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序异常检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>小目标检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像异常检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>文档场景信息抽取</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>



> ❗**注：上述能力是PaddleX基于GPU/CPU实现的能力，PaddleX还支持英伟达 GPU、昆仑芯、昇腾和寒武纪等多种主流硬件的无缝切换，但不同芯片支持的功能有所差异，以下列出了其他三种硬件支持的能力：**
<details>
  <summary>👉 昇腾芯能力支持</summary>

<table>
    <tr>
        <td></td>
        <td>在线体验</td>
        <td>快速推理</td>
        <td>高性能部署</td>
        <td>服务化部署</td>
        <td>端侧部署</td>
        <td>二次开发</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>目标检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>语义分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>实例分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>表格识别</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序预测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序异常检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>小目标检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像异常检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>文档场景信息抽取</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>
</details>


<details>
  <summary>👉 昆仑芯能力支持</summary>

<table>
    <tr>
        <td></td>
        <td>在线体验</td>
        <td>快速推理</td>
        <td>高性能部署</td>
        <td>服务化部署</td>
        <td>端侧部署</td>
        <td>二次开发</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>目标检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>语义分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>实例分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>表格识别</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序预测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序异常检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>小目标检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像异常检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>文档场景信息抽取</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>
</details>

<details>
  <summary>👉 寒武纪能力支持</summary>

<table class="centered-table">
    <tr>
        <td></td>
        <td>在线体验</td>
        <td>快速推理</td>
        <td>高性能部署</td>
        <td>服务化部署</td>
        <td>端侧部署</td>
        <td>二次开发</td>
        <td><a href = "https://aistudio.baidu.com/pipeline/mine">星河零代码产线</a></td>
    </tr>
    <tr>
        <td>OCR</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100061/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>目标检测</td>
        <td><a href = "hhttps://aistudio.baidu.com/community/app/70230/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>语义分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100062/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>实例分割</td>
        <td><a href = "https://aistudio.baidu.com/community/app/100063/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>表格识别</td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序预测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105706/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序异常检测</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105708/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>时序分类</td>
        <td><a href = "https://aistudio.baidu.com/community/app/105707/webUI?source=appMineRecent">体验链接</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像多标签分类</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>小目标检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>图像异常检测</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td>文档场景信息抽取</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
</table>

</details>





## ⏭️ 快速开始

### 🛠️ 安装

> ❗安装PaddleX前请先确保您有基础的Python运行环境，如果您还未安装Python环境，可以参考[运行环境准备](/docs_new/installation/installation.md#1-运行环境准备)进行安装

```bash
# 您的机器安装的是CUDA 11，请运行以下命令安装
pip install paddlepaddle-gpu
# 您的机器是CPU，请运行以下命令安装
pip install paddlepaddle
...
```
  
更多安装方式参考[PaddleX安装教程](/docs_new/installation/installation.md)

### 💻 命令行使用

一行命令即可快速体验产线效果，以通用OCR产线为例：
```bash
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
| 通用图像分类产线 | `image_classification` |   [通用图像分类产线命令行使用说明](/docs_new/pipelines_usage/tutorials/cv_pipelines/image_classification.md)   |
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
| 通用图像分类产线 | `image_classification` |   [通用图像分类产线Python脚本使用说明](/docs_new/pipelines_usage/tutorials/cv_pipelines/image_classification.md)   |
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

## 📖 文档
<details>
  <summary> <b> 安装 </b></summary>
  
  * [PaddleX安装教程](/docs_new/installation/installation.md) 
  * [PaddlePaddle安装教程](/docs_new/installation/paddlepaddle_install.md)

</details>

<details open>
<summary> <b> 产线使用 </b></summary>

* [PaddleX产线开发工具使用教程](/docs_new/pipelines_usage/pipeline_develop_guide.md)

* <details open>
    <summary> <b> cv相关产线 </b></summary>

   * [通用图像分类产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/image_classification.md)
   * [通用目标检测产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/object_detection.md)
   * [通用实例分割产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/instance_segmentation.md)
   * [通用语义分割产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/semantic_segmentation.md)
   * [通用图像多标签分类产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/image_multi_label_lassification.md)
   * [小目标检测产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/small_object_detection.md)
   * [图像异常检测产线](/docs_new/pipelines_usage/tutorials/cv_pipelines/image_anomaly_detection.md)
  </details>
  
* <details open>
    <summary> <b> ocr相关产线 </b></summary>

    * [通用OCR产线](/docs_new/pipelines_usage/tutorials/ocr_pipelies/OCR.md)
    * [通用表格识别产线](/docs_new/pipelines_usage/tutorials/ocr_pipelies/table_recognition.md)
  </details>

* <details open>
    <summary> <b> 时序相关产线</b> </summary>

   * [通用时序预测产线](/docs_new/pipelines_usage/tutorials/time_series_pipelines/time_series_forecasting.md)
   * [通用时序异常检测产线](/docs_new/pipelines_usage/tutorials/time_series_pipelines/time_series_anomaly_detection.md)
   * [通用时序分类产线](/docs_new/pipelines_usage/tutorials/time_series_pipelines/time_series_classification.md)
  </details>

* <details open>
    <summary> <b> 信息抽取相关产线 </b></summary>

   * [文档场景信息抽取产线](/docs_new/pipelines_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md)
  </details>

* <details>
    <summary> <b> 相关说明文件</b> </summary>

   * [PaddleX产线命令行使用说明](/docs_new/pipelines_usage/instructions/pipeline_CLI_usage.md)
   * [PaddleX产线Python脚本使用说明](/docs_new/pipelines_usage/instructions/pipeline_python_API.md)
  </details>
   
</details>

<details>
<summary> <b> 单功能模块使用 </b></summary>

* <details>
  <summary> <b> cv相关模块 </b></summary>
  
  * [图像分类模块]()
  * [图像识别模块]()
  * [目标检测模块]()
  * [小目标检测模块]()
  * [人脸检测模块]()
  * [主体检测模块]()
  * [行人检测模块]()
  * [车辆检测模块]()
  * [语义分割模块]()
  * [实例分割模块]()
  * [文档图像方向分类]()
  * [图像多标签分类]()
  * [行人属性识别]()
  * [车辆属性识别]()
  * [图像矫正模块]()
  * [无监督异常检测模块]()
  </details> 
  
* <details>
  <summary> <b> ocr相关模块 </b></summary>

  * [文本检测模块]()
  * [印章文本检测模块]()
  * [文本识别模块]()
  * [版面区域定位模块]()
  * [表格结构识别模块]()
  </details>

* <details>
  <summary> <b> 时序相关产线 </b></summary>

  * [时序预测模块]()
  * [时序异常检测模块]()
  * [时序分类模块]()
  </details>
    
* <details>
  <summary> <b> 相关说明文件 </b></summary>

  * [PaddleX单模型Python脚本使用说明]()
  * [PaddleX通用模型配置文件参数说明]()
  * [PaddleX时序任务模型配置文件参数说明]()
  </details>

</details>

<details>
  <summary> <b> 多模块组合使用 </b></summary>

  * [多功能模块组合使用教程]()
</details>
<details>
  <summary> <b> 模型产线部署 </b></summary>

  * [PaddleX 服务化部署指南]()
  * [PaddleX端侧部署指南]()
  * [PaddleX 高性能部署指南]()
</details>
<details>
  <summary> <b> 多硬件使用 </b></summary>

  * [多硬件使用指南]()
</details>

## 🔥 模型产线列表

<details>
  <summary><a herf = ""> <b> 文档场景信息抽取产线 </b></a></summary>
</details>

<details>
  <summary> <a herf = ""><b>通用OCR产线  </b></a></summary>

| 任务模块 | 模型            | 精度  | GPU推理耗时（ms） | CPU推理耗时 | 模型存储大小（M) | 
|----------|---------------------|-------|-------------------|-------------|------------------|
| 文本检测 | PP-OCRv4_mobile_det | 77.79 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_det | 82.69 | 22.20346          | 2662.158    | 198              | 
| 文本识别 | PP-OCRv4_mobile_rec | 78.20 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_rec | 79.20 | 22.20346          | 2662.158    | 198              | 

**注：文本检测模型精度指标为 Hmean(%)，文本识别模型精度指标为 Accuracy(%)。**

</details>

<details>
  <summary><a herf = ""> <b> 通用表格识别产线 </b> </a></summary>
</details>

<details>
  <summary><a herf = ""> <b> 通用图像分类产线  </b></a></summary>

| 任务模块 | 模型            | 精度  | GPU推理耗时（ms） | CPU推理耗时 | 模型存储大小（M) | 
|----------|---------------------|-------|-------------------|-------------|------------------|
| 文本检测 | PP-OCRv4_mobile_det | 77.79 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_det | 82.69 | 22.20346          | 2662.158    | 198              | 
| 文本识别 | PP-OCRv4_mobile_rec | 78.20 | 2.719474          | 79.1097     | 15               | 
|          | PP-OCRv4_server_rec | 79.20 | 22.20346          | 2662.158    | 198              | 

**注：文本检测模型精度指标为 Hmean(%)，文本识别模型精度指标为 Accuracy(%)。**

</details>

<details>
  <summary><a herf = ""> <b> 通用目标检测产线  </b></a></summary>
</details>

<details>
  <summary><a herf = ""> <b> 通用实例分割产线  </b></a></summary>
</details>

<details>
  <summary> <a herf = ""><b> 通用语义分割产线 </b> </a></summary>
</details>

<details>
  <summary> <a herf = ""><b> 通用图像多标签分类产线 </b> </a></summary>
</details>

<details>
  <summary><a herf = ""> <b> 小目标检测分类产线 </b> </a></summary>
</details>


<details>
  <summary><a herf = ""> <b> 图像异常检测产线 </b> </a></summary>
</details>


<details>
  <summary><a herf = ""> <b> 通用时序预测产线 </b> </a></summary>
</details>

<details>
  <summary><a herf = ""> <b> 通用时序异常检测产线 </b> </a></summary>
</details>

<details>
  <summary><a herf = ""> <b> 通用时序分类产线</b>  </a></summary>
</details>



## 🤔 FAQ

关于我们项目的一些常见问题解答，请参考[FAQ](/docs_new/FAQ.md)。如果您的问题没有得到解答，请随时在 [Issues](https://github.com/PaddlePaddle/PaddleX/issues) 中提出
## 💬 Discussion

我们非常欢迎并鼓励社区成员在[Discussions](https://github.com/PaddlePaddle/PaddleX/discussions)板块中提出问题、分享想法和反馈。无论您是想要报告一个bug、讨论一个功能请求、寻求帮助还是仅仅想要了解项目的最新动态，这里都是一个绝佳的平台。
## 📄 许可证书

本项目的发布受[Apache 2.0 license](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta/LICENSE)许可认证。