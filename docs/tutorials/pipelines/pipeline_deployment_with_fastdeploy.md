# 基于FastDeploy的模型产线部署

除了[模型产线推理 Python API 文档](./pipeline_inference_api.md)中介绍的模型产线部署方案外，PaddleX还提供基于全场景、多后端推理工具FastDeploy的部署方案。基于FastDeploy的部署方案支持更多后端，并且提供高性能推理和服务化部署两种部署方式，能够满足更多场景的需求：

- **高性能推理**：运行脚本执行推理，或在程序中调用Python/C++的推理API。旨在实现测试样本的高效输入与模型预测结果的快速获取，特别适用于大规模批量刷库的场景，显著提升数据处理效率。
- **服务化部署**：采用C/S架构，以服务形式提供推理能力，客户端可以通过网络请求访问服务，以获取推理结果。

请注意，目前并非所有模型都支持基于FastDeploy的部署，具体支持情况请参考[模型部署支持情况](#模型部署支持情况)。

## 操作流程

1. 获取离线部署包。
    1. 在[星河社区](https://aistudio.baidu.com/pipeline/mine)创建产线，在“选择产线”页面点击“直接部署”。
    2. 在“产线部署”页面选择“导出离线部署包”，使用默认的模型方案，点击“导出部署包”。
    3. 待部署包导出完毕后，点击“下载离线部署包”，将部署包下载到本地。
    4. 点击“生成部署包序列号”，根据页面提示完成设备指纹的获取以及设备指纹与序列号的绑定，确保序列号对应的激活状态为“已激活“。
2. 使用自训练模型替换离线部署包`model`目录中的模型。需注意模型与产线模块的对应关系，并且不要修改`model`的目录结构。
3. 根据需要选择要使用的部署SDK：`offline_sdk`目录对应高性能推理SDK，`serving_sdk`目录对应服务化部署SDK。按照SDK文档（`README.md`）中的说明，完成产线的本地部署。

## 模型部署支持情况

| 模型名称 | 是否支持基于FastDeploy的部署 |
| :---: | :---: |
| ResNet18 | 是 |
| ResNet34 | 是 |
| ResNet50 | 是 |
| ResNet101 | 是 |
| ResNet152 | 是 |
| ResNet18 | 是 |
| PP-LCNet_x0_25 | 是 |
| PP-LCNet_x0_35 | 是 |
| PP-LCNet_x0_5 | 是 |
| PP-LCNet_x0_75 | 是 |
| PP-LCNet_x1_0 | 是 |
| PP-LCNet_x1_5 | 是 |
| PP-LCNet_x2_5 | 是 |
| PP-LCNet_x2_0 | 是 |
| MobileNetV3_large_x0_35 | 是 |
| MobileNetV3_large_x0_5 | 是 |
| MobileNetV3_large_x0_75 | 是 |
| MobileNetV3_large_x1_0 | 是 |
| MobileNetV3_large_x1_25 | 是 |
| MobileNetV3_small_x0_35 | 是 |
| MobileNetV3_small_x0_5 | 是 |
| MobileNetV3_small_x0_75 | 是 |
| MobileNetV3_small_x1_0 | 是 |
| MobileNetV3_small_x1_25 | 是 |
| ConvNeXt_tiny | 是 |
| MobileNetV2_x0_25 | 是 |
| MobileNetV2_x0_5 | 是 |
| MobileNetV2_x1_0 | 是 |
| MobileNetV2_x1_5 | 是 |
| MobileNetV2_x2_0 | 是 |
| SwinTransformer_base_patch4_window7_224 | 否 |
| PP-HGNet_small | 是 |
| PP-HGNetV2-B0 | 是 |
| PP-HGNetV2-B4 | 是 |
| PP-HGNetV2-B6 | 是 |
| CLIP_vit_base_patch16_224 | 是 |
| CLIP_vit_large_patch14_224 | 是 |
| PP-YOLOE_plus-X | 是 |
| PP-YOLOE_plus-L | 是 |
| PP-YOLOE_plus-M | 是 |
| PP-YOLOE_plus-S | 是 |
| RT-DETR-L | 否 |
| RT-DETR-H | 否 |
| RT-DETR-X | 否 |
| RT-DETR-R18 | 否 |
| RT-DETR-R50 | 否 |
| PicoDet-S | 是 |
| PicoDet-L | 是 |
| Deeplabv3-R50 | 是 |
| Deeplabv3-R101 | 是 |
| Deeplabv3_Plus-R50 | 是 |
| Deeplabv3_Plus-R101 | 是 |
| PP-LiteSeg-T | 是 |
| OCRNet_HRNet-W48 | 是 |
| Mask-RT-DETR-H | 否 |
| Mask-RT-DETR-L | 否 |
| PP-OCRv4_server_rec | 是 |
| PP-OCRv4_mobile_rec | 是 |
| PP-OCRv4_server_det | 是 |
| PP-OCRv4_mobile_det | 是 |
| PicoDet_layout_1x | 是 |
| SLANet | 否 |
