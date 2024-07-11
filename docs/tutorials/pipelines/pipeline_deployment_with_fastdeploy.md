# 基于 FastDeploy 的模型产线部署

除了 [模型产线推理 Python API 文档](./pipeline_inference_api.md) 中介绍的模型产线部署方案外，PaddleX 还提供基于全场景、多后端推理工具 FastDeploy 的部署方案。基于 FastDeploy 的部署方案支持更多后端，并且提供高性能推理和服务化部署两种部署方式，能够满足更多场景的需求：

- **高性能推理**：运行脚本执行推理，或在程序中调用 Python/C++ 的推理 API。旨在实现测试样本的高效输入与模型预测结果的快速获取，特别适用于大规模批量刷库的场景，显著提升数据处理效率。
- **服务化部署**：采用 C/S 架构，以服务形式提供推理能力，客户端可以通过网络请求访问服务，以获取推理结果。

请注意，目前并非所有产线、所有模型都支持基于 FastDeploy 的部署，具体支持情况请参考 [支持 FastDeploy 部署的产线与模型](#支持-FastDeploy-部署的产线与模型)。

## 操作流程

1. 获取离线部署包。
    1. 在 [AIStudio 星河社区](https://aistudio.baidu.com/pipeline/mine) 创建产线，在“选择产线”页面点击“直接部署”。
    2. 在“产线部署”页面选择“导出离线部署包”，使用默认的模型方案，点击“导出部署包”。
    3. 待部署包导出完毕后，点击“下载离线部署包”，将部署包下载到本地。
    4. 点击“生成部署包序列号”，根据页面提示完成设备指纹的获取以及设备指纹与序列号的绑定，确保序列号对应的激活状态为“已激活“。
2. 使用自训练模型替换离线部署包 `model` 目录中的模型。需注意模型与产线模块的对应关系，并且不要修改 `model` 的目录结构。
3. 根据需要选择要使用的部署SDK：`offline_sdk` 目录对应高性能推理SDK，`serving_sdk` 目录对应服务化部署SDK。按照SDK文档（`README.md`）中的说明，完成产线的本地部署。

## 支持 FastDeploy 部署的产线与模型

<table>
    <tr>
        <th>模型产线</th>
        <th>产线模块</th>
        <th>具体模型</th>
    </tr>
    <tr>
        <td>通用图像分类</td>
        <td>图像分类</td>
        <td>CLIP_vit_base_patch16_224<br/>CLIP_vit_large_patch14_224<details>
        <summary><b>more</b></summary><br/>ConvNeXt_tiny<br/>MobileNetV2_x0_25<br/>MobileNetV2_x0_5<br/>MobileNetV2_x1_0<br/>MobileNetV2_x1_5<br/>MobileNetV2_x2_0<br/>MobileNetV3_large_x0_35<br/>MobileNetV3_large_x0_5<br/>MobileNetV3_large_x0_75<br/>MobileNetV3_large_x1_0<br/>MobileNetV3_large_x1_25<br/>MobileNetV3_small_x0_35<br/>MobileNetV3_small_x0_5<br/>MobileNetV3_small_x0_75<br/>MobileNetV3_small_x1_0<br/>MobileNetV3_small_x1_25<br/>PP-HGNet_small<br/>PP-HGNetV2-B0<br/>PP-HGNetV2-B4<br/>PP-HGNetV2-B6<br/>PP-LCNet_x0_25<br/>PP-LCNet_x0_35<br/>PP-LCNet_x0_5<br/>PP-LCNet_x0_75<br/>PP-LCNet_x1_0<br/>PP-LCNet_x1_5<br/>PP-LCNet_x2_0<br/>PP-LCNet_x2_5<br/>ResNet18<br/>ResNet34<br/>ResNet50<br/>ResNet101<br/>ResNet152</details></td>
    </tr>
    <tr>
        <td>通用目标检测</td>
        <td>目标检测</td>
        <td>PicoDet-S<br/>PicoDet-L<details>
        <summary><b>more</b></summary><br/>PP-YOLOE_plus-S<br/>PP-YOLOE_plus-M<br/>PP-YOLOE_plus-L<br/>PP-YOLOE_plus-X</details></td>
    </tr>
    <tr>
        <td>通用语义分割</td>
        <td>语义分割</td>
        <td>OCRNet_HRNet-W48<br/>PP-LiteSeg-T<details>
        <summary><b>more</b></summary><br/>Deeplabv3-R50<br/>Deeplabv3-R101<br/>Deeplabv3_Plus-R50<br/>Deeplabv3_Plus-R101</details></td>
    </tr>
    <tr>
        <td rowspan="2">通用 OCR</td>
        <td>文本检测</td>
        <td>PP-OCRv4_mobile_det<br/>PP-OCRv4_server_det</td>
    </tr>
    <tr>
        <td>文本识别</td>
        <td>PP-OCRv4_mobile_rec<br/>PP-OCRv4_server_rec</td>
    </tr>
    </tr>
</table>
