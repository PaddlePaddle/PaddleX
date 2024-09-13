<p align="center">
  <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/63c6d059-234f-4a27-955e-ac89d81409ee" width="360" height ="55" alt="PaddleX" align="middle" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-red.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.8%2C%203.9%2C%203.10-blue.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20windows-orange.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/hardware-intel cpu%2C%20gpu%2C%20xpu%2C%20npu%2C%20mlu-yellow.svg"></a>
</p>

## 简介
PaddleX 3.0 是飞桨精选模型的低代码开发工具，支持国内外多款主流硬件的模型训练和推理，覆盖工业、能源、金融、交通、教育等全行业，助力开发者产业实践落地。

|                **通用图像分类**                 |                **通用目标检测**                 |                **通用语义分割**                 |                **通用实例分割**                 |
| :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: | :-------------------------------------------: |
|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/b302cd7e-e027-4ea6-86d0-8a4dd6d61f39" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/099e2b00-0bbe-4b20-9c5a-96b69e473bd2" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/02637f8c-f248-415b-89ab-1276505f198c" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/09f683b4-27df-4c24-b8a7-84da20fdd182" height="126px" width="180px">|
|                  **通用OCR**                   |                **通用表格识别**                 |               **通用场景信息抽取**               |               **文档场景信息抽取**               |
|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1ef48536-48d4-484b-a6fb-0d6631ba2386" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/1e798e05-dee7-4b41-9cc4-6708b6014efa" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/04218629-4a7b-48ea-b815-977a05fbbb13" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/e3d97f4e-ab46-411c-8155-494c61492b0a" height="126px" width="180px">|
|                  **时序预测**                   |                **时序异常检测**                 |                 **时序分类**                   |              **多模型融合时序预测**              |
|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/6e897bf6-35fe-45e6-a040-e9a1a20cfdf2" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/c54c66cc-da4f-4631-877b-43b0fbb192a6" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0ce925b2-3776-4dde-8ce0-5156d5a2476e" height="126px" width="180px">|<img src="https://github.com/PaddlePaddle/PaddleX/assets/142379845/0959d099-a17c-40bc-9c2b-13f4f5e24ddc" height="126px" width="180px">|




## 📣 近期更新
- 🔥🔥**直播和实战打卡营预告：** 《PaddleX 3.0 Beta 重磅开源：多场景低代码AI开发，本地多硬件全兼容》课程上线，分享 PaddleX 3.0 Beta 版本新特色及全新开发范式，详解基于真实产业用户场景与业务数据，如何利用本地GPU算力，低成本零门槛解决产业实际问题。**直播时间：7月16日（周二）19：00。**[报名链接](https://www.wjx.top/vm/rXqxgT5.aspx?udsid=875333)。
- 🔥 **2024.6.27，PaddleX 3.0 Beta 本地端正式发布，支持以低代码的方式在本地端使用多种主流硬件进行产线和模型开发。**
- 🔥 **2024.3.25，PaddleX 3.0 云端发布，支持在[AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine)以零代码的方式【创建产线】使用。**

## 🌟 特性

PaddleX 3.0 集成了飞桨生态的优势能力，覆盖 7 大场景任务，构建了 16 条模型产线，提供低代码开发模式，助力开发者在多种主流硬件上实现模型全流程开发。

- **基础模型产线（模型丰富，场景全面）：** 精选 68 个优质飞桨模型，涵盖图像分类、目标检测、图像分割、OCR、文本图像版面分析、文本图像信息抽取、时序分析任务场景。
- **特色模型产线（显著提升效率）：** 提供大小模型结合、大模型半监督学习和多模型融合的高效解决方案。
- **低门槛开发模式（便捷开发与部署）：** 提供零代码和低代码两种开发方式。
  - **零代码开发：** 用户通过图形界面（GUI）交互式提交后台训练任务，打通在线和离线部署，并支持以 API 形式调用在线服务。
  - **低代码开发：** 通过统一的 API 接口实现 16 条模型产线的全流程开发，同时支持用户自定义模型流程串联。
- **多硬件本地支持（兼容性强）：** 支持英伟达 GPU、昆仑芯、昇腾和寒武纪等多种硬件，纯离线使用。

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/61c4738f-735e-4ceb-aa5f-1038d4506d1c">
</div>

## ⚡ 安装与快速开始
- [安装](./docs/tutorials/INSTALL.md)
- 快速开始
  - [模型产线开发工具](./docs/tutorials/pipelines/pipeline_develop_tools.md)
  - [单模型开发工具](./docs/tutorials/models/model_develop_tools.md)

## 🛠️ PaddleX 3.0 覆盖的模型和模型产线
PaddleX 3.0 覆盖了 16 条产业级模型产线，其中 9 条基础产线可以直接使用本仓库离线使用，7 条特色产线可在飞桨 [AI Studio 星河社区](https://aistudio.baidu.com/pipeline/mine)上免费使用。
<table>
  <tr>
    <th>产线类型</th>
    <th>模型产线</th>
    <th>产线模块</th>
    <th>具体模型</th>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>通用图像分类</td>
    <td>图像分类</td>
    <td>CLIP_vit_base_patch16_224<br/>CLIP_vit_large_patch14_224<details>
    <summary><b>more</b></summary><br/>ConvNeXt_tiny<br/>ConvNeXt_small<br/>ConvNeXt_base_224<br/>ConvNeXt_base_384<br/>ConvNeXt_large_224<br/>ConvNeXt_large_384<br/>MobileNetV1_x0_25<br/>MobileNetV1_x0_5<br/>MobileNetV1_x0_75<br/>MobileNetV1_x1_0<br/>MobileNetV2_x0_25<br/>MobileNetV2_x0_5<br/>MobileNetV2_x1_0<br/>MobileNetV2_x1_5<br/>MobileNetV2_x2_0<br/>MobileNetV3_large_x0_35<br/>MobileNetV3_large_x0_5<br/>MobileNetV3_large_x0_75<br/>MobileNetV3_large_x1_0<br/>MobileNetV3_large_x1_25<br/>MobileNetV3_small_x0_35<br/>MobileNetV3_small_x0_5<br/>MobileNetV3_small_x0_75<br/>MobileNetV3_small_x1_0<br/>MobileNetV3_small_x1_25<br/>PP-HGNet_tiny<br/>PP-HGNet_small<br/>PP-HGNet_base<br/>PP-HGNetV2-B0<br/>PP-HGNetV2-B1<br/>PP-HGNetV2-B2<br/>PP-HGNetV2-B3<br/>PP-HGNetV2-B4<br/>PP-HGNetV2-B5<br/>PP-HGNetV2-B6<br/>PP-LCNet_x0_25<br/>PP-LCNet_x0_35<br/>PP-LCNet_x0_5<br/>PP-LCNet_x0_75<br/>PP-LCNet_x1_0<br/>PP-LCNet_x1_5<br/>PP-LCNet_x2_0<br/>PP-LCNet_x2_5<br/>PP-LCNetV2_small<br/>PP-LCNetV2_base<br/>PP-LCNetV2_large<br/>ResNet18<br/>ResNet18_vd<br/>ResNet34<br/>ResNet34_vd<br/>ResNet50<br/>ResNet50_vd<br/>ResNet101<br/>ResNet101_vd<br/>ResNet152<br/>ResNet152_vd<br/>ResNet200_vd<br/>SwinTransformer_tiny_patch4_window7_224<br/>SwinTransformer_small_patch4_window7_224<br/>SwinTransformer_base_patch4_window7_224<br/>SwinTransformer_base_patch4_window12_384<br/>SwinTransformer_large_patch4_window7_224<br/>SwinTransformer_large_patch4_window12_384</details></td>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>通用目标检测</td>
    <td>目标检测</td>
    <td>PicoDet-S<br/>PicoDet-L<details>
    <summary><b>more</b></summary><br/>PP-YOLOE_plus-S<br/>PP-YOLOE_plus-M<br/>PP-YOLOE_plus-L<br/>PP-YOLOE_plus-X<br/>RT-DETR-L<br/>RT-DETR-H<br/>RT-DETR-X<br/>RT-DETR-R18<br/>RT-DETR-R50<br/>YOLOv3-DarkNet53<br/>YOLOv3-MobileNetV3<br/>YOLOv3-ResNet50_vd_DCN<br/>YOLOX-L<br/>YOLOX-M<br/>YOLOX-N<br/>YOLOX-S<br/>YOLOX-T<br/>YOLOX-X</details></td>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>通用语义分割</td>
    <td>语义分割</td>
    <td>OCRNet_HRNet-W48<br/>OCRNet_HRNet-W18<br/>PP-LiteSeg-T<details>
    <summary><b>more</b></summary><br/>Deeplabv3-R50<br/>Deeplabv3-R101<br/>Deeplabv3_Plus-R50<br/>Deeplabv3_Plus-R101<br/>SeaFormer_tiny<br/
    >SeaFormer_small<br/>SeaFormer_base<br/>SeaFormer_large<br/
    >SegFormer-B0<br/>SegFormer-B1<br/>SegFormer-B2<br/
    >SegFormer-B3<br/>SegFormer-B4<br/>SegFormer-B5</details></td>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>通用实例分割</td>
    <td>实例分割</td>
    <td>Mask-RT-DETR-L<br/>Mask-RT-DETR-H</td>
  </tr>
  <tr>
    <td rowspan="3">基础产线</td>
    <td rowspan="3">通用OCR</td>
    <td>文本检测</td>
    <td>PP-OCRv4_mobile_det<br/>PP-OCRv4_server_det</td>
  </tr>
  <tr>
    <td>文本识别</td>
    <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
  </tr>
  <tr>
    <td>公式识别</td>
    <td>LaTeX_OCR_rec</td>
  </tr>
  <tr>
    <td rowspan="4">基础产线</td>
    <td rowspan="4">通用表格识别</td>
    <td>版面区域检测</td>
    <td>PicoDet layout_1x<br/>PicoDet-L_layout<br/>RT-DETR-H_layout_3cls<br/>RT-DETR-H_layout_17cls</td>
  </tr>
  <tr>
    <td>表格识别</td>
    <td>SLANet</td>
  </tr>
  <tr>
    <td>文本检测</td>
    <td>PP-OCRv4_mobile_det<br/>PP-OCRv4_server_det</td>
  </tr>
  <tr>
    <td>文本识别</td>
    <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>时序预测</td>
    <td>时序预测</td>
    <td>DLinear<br/>Nonstationary<br/>TiDE<br/>PatchTST<br/>TimesNet</td>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>时序异常检测</td>
    <td>时序异常检测</td>
    <td>DLinear_ad<br/>Nonstationary_ad<br/>AutoEncoder_ad<br/>PatchTST_ad<br/>TimesNet_ad</td>
  </tr>
  <tr>
    <td>基础产线</td>
    <td>时序分类</td>
    <td>时序分类</td>
    <td>TimesNet_cls</td>
  </tr>
 <tr>
    <td>特色产线</td>
    <td>大模型半监督学习-图像分类</td>
    <td>大模型半监督学习-图像分类</td>
    <td>CLIP_vit_base_patch16_224<br/>MobileNetV3_small_x1_0<br/><details><summary><b>more</b></summary>PP-HGNet_small<br/>PP-HGNetV2-B0<br/>PP-HGNetV2-B4<br/>PP-HGNetV2-B6<br/>PP-LCNet_x1_0<br/>ResNet50<br/>SwinTransformer_base_patch4_window7_224</details></td>
  </tr>
  <tr>
    <td>特色产线</td>
    <td>大模型半监督学习-目标检测</td>
    <td>大模型半监督学习-目标检测</td>
    <td>PicoDet-S<br/>PicoDet-L<details>
    <summary><b>more</b></summary><br/>PP-YOLOE plus-S<br/>PP-YOLOE_plus-L<br/>RT-DETR-H</details></td>
  </tr>
  <tr>
    <td rowspan="2">特色产线</td>
    <td rowspan="2">大模型半监督学习-OCR</td>
    <td>文本检测</td>
    <td>PP-OCRv4_mobile_det<br/>PP-OCRv4_server_det</td>
  </tr>
  <tr>
    <td>大模型半监督学习-文本识别</td>
    <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
   </tr>
<tr>
    <td rowspan="3">特色产线</td>
    <td rowspan="3">通用场景信息抽取v2<br>(PP-ChatOCRv2-common)</td>
    <td>文本识别</td>
    <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
  </tr>
  <tr>
    <td>文本检测</td>
    <td>PP-OCRv4_mobile_det<br/>PP-OCRv4_server_det</td>
  </tr>
  <tr>
    <td>prompt工程</td>
    <td>-</td>
  </tr>
  <tr>
    <td rowspan="5">特色产线</td>
    <td rowspan="5">文档场景信息抽取v2<br>(PP-ChatOCRv2-doc)</td>
    <td>版面分析</td>
    <td>PicoDet layout_1x</td>
  </tr>
  <tr>
    <td>文本检测</td>
    <td>PP-OCRv4_mobile_det<br/>PP-OCRv4_server_det</td>
  </tr>
  <tr>
    <td>文本识别</td>
    <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
  </tr>
  <tr>
    <td>表格识别</td>
    <td>SLANet</td>
  </tr>
  <tr>
    <td>prompt工程</td>
    <td>-</td>
  </tr>
  <tr>
    <td>特色产线</td>
    <td>多模型融合时序预测v2<br>(PP-TSv2_forecast)</td>
    <td>时序预测</td>
    <td>多模型融合时序预测</td>
  </tr>
  <tr>
    <td>特色产线</td>
    <td>多模型融合时序异常检测v2<br>(PP-TSv2_anomaly)</td>
    <td>时序异常检测</td>
    <td>多模型融合时序异常检测</td>
  </tr>
</table>


  - [模型产线列表](./docs/tutorials/pipelines/support_pipeline_list.md)
  - [单模型列表](./docs/tutorials/models/support_model_list.md)

## 📖 零代码开发教程
<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleX/assets/45199522/f3238aae-76e3-4b25-8e4f-238fb6096bf8">
</div>

- [云端图形化开发界面](https://aistudio.baidu.com/pipeline/mine)：支持开发者使用零代码产线产出高质量模型和部署包。
- [教程《零门槛开发产业级 AI 模型》](https://aistudio.baidu.com/practical/introduce/546656605663301)：提供产业级模型开发经验，并且用 12 个实用的产业实践案例，手把手带你零门槛开发产业级 AI 模型。

## 📖 低代码开发教程

### 一、模型产线开发工具 🔥
PaddleX 3.0 模型产线开发工具支持开发者通过 6 个步骤，完成产业级落地解决方案的开发。PaddleX 3.0 支持的模型产线可以参考 [PaddleX 模型产线列表](./docs/tutorials/pipelines/support_pipeline_list.md)。
- [模型产线开发流程](./docs/tutorials/pipelines/pipeline_develop_tools.md)
- [模型产线推理预测](./docs/tutorials/pipelines/pipeline_inference.md)
- [产线模型选型](./docs/tutorials/pipelines/model_select.md)


### 二、单模型开发工具 🚀
PaddleX 3.0 单模型开发工具支持开发者以低代码的方式快速实现模型的开发和优化，包括数据准备、模型训练/评估、模型推理的使用方法，方便低成本集成到模型产线中。PaddleX3.0 支持的模型可以参考 [PaddleX 模型库](./docs/tutorials/models/support_model_list.md)。
- [数据校验](./docs/tutorials/data/dataset_check.md)
- [模型开发](./docs/tutorials/models/model_develop_tools.md)

## 🌟 多硬件支持
PaddleX 3.0 支持在多种硬件上进行模型的开发，除了 GPU 外，当前支持的硬件还有**昆仑芯**、**昇腾**、**寒武纪**。只需添加一个配置设备的参数，即可在对应硬件上使用上述工具。使用方式详情[多硬件使用](./docs/tutorials/base/devices_use_guidance.md)。

- 昇腾芯支持的模型列表请参考 [PaddleX 昇腾模型列表](./docs/tutorials/models/support_npu_model_list.md)。
- 昆仑芯支持的模型列表请参考 [PaddleX 昆仑芯模型列表](./docs/tutorials/models/support_xpu_model_list.md)。
- 寒武纪芯支持的模型列表请参考 [PaddleX 寒武纪模型列表](./docs/tutorials/models/support_mlu_model_list.md)。


## 👀 贡献代码

我们非常欢迎您为 PaddleX 贡献代码或者提供使用建议。如果您可以修复某个 issue 或者增加一个新功能，欢迎给我们提交 Pull Requests。

## 许可证书
本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。
