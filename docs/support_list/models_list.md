简体中文 | [English](models_list_en.md)

# PaddleX模型列表（CPU/GPU）

PaddleX 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## [图像分类模块](../module_usage/tutorials/cv_modules/image_classification.md)
|模型名称|Top1 Acc（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|CLIP_vit_base_patch16_224|85.36|13.1957|285.493|306.5 M|[CLIP_vit_base_patch16_224.yaml](../../paddlex/configs/image_classification/CLIP_vit_base_patch16_224.yaml)|
|CLIP_vit_large_patch14_224|88.1|51.1284|1131.28|1.04 G|[CLIP_vit_large_patch14_224.yaml](../../paddlex/configs/image_classification/CLIP_vit_large_patch14_224.yaml)|
|ConvNeXt_base_224|83.84|12.8473|1513.87|313.9 M|[ConvNeXt_base_224.yaml](../../paddlex/configs/image_classification/ConvNeXt_base_224.yaml)|
|ConvNeXt_base_384|84.90|31.7607|3967.05|313.9 M|[ConvNeXt_base_384.yaml](../../paddlex/configs/image_classification/ConvNeXt_base_384.yaml)|
|ConvNeXt_large_224|84.26|26.8103|2463.56|700.7 M|[ConvNeXt_large_224.yaml](../../paddlex/configs/image_classification/ConvNeXt_large_224.yaml)|
|ConvNeXt_large_384|85.27|66.4058|6598.92|700.7 M|[ConvNeXt_large_384.yaml](../../paddlex/configs/image_classification/ConvNeXt_large_384.yaml)|
|ConvNeXt_small|83.13|9.74075|1127.6|178.0 M|[ConvNeXt_small.yaml](../../paddlex/configs/image_classification/ConvNeXt_small.yaml)|
|ConvNeXt_tiny|82.03|5.48923|672.559|101.4 M|[ConvNeXt_tiny.yaml](../../paddlex/configs/image_classification/ConvNeXt_tiny.yaml)|
|FasterNet-L|83.5|23.4415|-|357.1 M|[FasterNet-L.yaml](../../paddlex/configs/image_classification/FasterNet-L.yaml)|
|FasterNet-M|83.0|21.8936|-|204.6 M|[FasterNet-M.yaml](../../paddlex/configs/image_classification/FasterNet-M.yaml)|
|FasterNet-S|81.3|13.0409|-|119.3 M|[FasterNet-S.yaml](../../paddlex/configs/image_classification/FasterNet-S.yaml)|
|FasterNet-T0|71.9|12.2432|-|15.1 M|[FasterNet-T0.yaml](../../paddlex/configs/image_classification/FasterNet-T0.yaml)|
|FasterNet-T1|75.9|11.3562|-|29.2 M|[FasterNet-T1.yaml](../../paddlex/configs/image_classification/FasterNet-T1.yaml)|
|FasterNet-T2|79.1|10.703|-|57.4 M|[FasterNet-T2.yaml](../../paddlex/configs/image_classification/FasterNet-T2.yaml)|
|MobileNetV1_x0_5|63.5|1.86754|7.48297|4.8 M|[MobileNetV1_x0_5.yaml](../../paddlex/configs/image_classification/MobileNetV1_x0_5.yaml)|
|MobileNetV1_x0_25|51.4|1.83478|4.83674|1.8 M|[MobileNetV1_x0_25.yaml](../../paddlex/configs/image_classification/MobileNetV1_x0_25.yaml)|
|MobileNetV1_x0_75|68.8|2.57903|10.6343|9.3 M|[MobileNetV1_x0_75.yaml](../../paddlex/configs/image_classification/MobileNetV1_x0_75.yaml)|
|MobileNetV1_x1_0|71.0|2.78781|13.98|15.2 M|[MobileNetV1_x1_0.yaml](../../paddlex/configs/image_classification/MobileNetV1_x1_0.yaml)|
|MobileNetV2_x0_5|65.0|4.94234|11.1629|7.1 M|[MobileNetV2_x0_5.yaml](../../paddlex/configs/image_classification/MobileNetV2_x0_5.yaml)|
|MobileNetV2_x0_25|53.2|4.50856|9.40991|5.5 M|[MobileNetV2_x0_25.yaml](../../paddlex/configs/image_classification/MobileNetV2_x0_25.yaml)|
|MobileNetV2_x1_0|72.2|6.12159|16.0442|12.6 M|[MobileNetV2_x1_0.yaml](../../paddlex/configs/image_classification/MobileNetV2_x1_0.yaml)|
|MobileNetV2_x1_5|74.1|6.28385|22.5129|25.0 M|[MobileNetV2_x1_5.yaml](../../paddlex/configs/image_classification/MobileNetV2_x1_5.yaml)|
|MobileNetV2_x2_0|75.2|6.12888|30.8612|41.2 M|[MobileNetV2_x2_0.yaml](../../paddlex/configs/image_classification/MobileNetV2_x2_0.yaml)|
|MobileNetV3_large_x0_5|69.2|6.31302|14.5588|9.6 M|[MobileNetV3_large_x0_5.yaml](../../paddlex/configs/image_classification/MobileNetV3_large_x0_5.yaml)|
|MobileNetV3_large_x0_35|64.3|5.76207|13.9041|7.5 M|[MobileNetV3_large_x0_35.yaml](../../paddlex/configs/image_classification/MobileNetV3_large_x0_35.yaml)|
|MobileNetV3_large_x0_75|73.1|8.41737|16.9506|14.0 M|[MobileNetV3_large_x0_75.yaml](../../paddlex/configs/image_classification/MobileNetV3_large_x0_75.yaml)|
|MobileNetV3_large_x1_0|75.3|8.64112|19.1614|19.5 M|[MobileNetV3_large_x1_0.yaml](../../paddlex/configs/image_classification/MobileNetV3_large_x1_0.yaml)|
|MobileNetV3_large_x1_25|76.4|8.73358|22.1296|26.5 M|[MobileNetV3_large_x1_25.yaml](../../paddlex/configs/image_classification/MobileNetV3_large_x1_25.yaml)|
|MobileNetV3_small_x0_5|59.2|5.16721|11.2688|6.8 M|[MobileNetV3_small_x0_5.yaml](../../paddlex/configs/image_classification/MobileNetV3_small_x0_5.yaml)|
|MobileNetV3_small_x0_35|53.0|5.22053|11.0055|6.0 M|[MobileNetV3_small_x0_35.yaml](../../paddlex/configs/image_classification/MobileNetV3_small_x0_35.yaml)|
|MobileNetV3_small_x0_75|66.0|5.39831|12.8313|8.5 M|[MobileNetV3_small_x0_75.yaml](../../paddlex/configs/image_classification/MobileNetV3_small_x0_75.yaml)|
|MobileNetV3_small_x1_0|68.2|6.00993|12.9598|10.5 M|[MobileNetV3_small_x1_0.yaml](../../paddlex/configs/image_classification/MobileNetV3_small_x1_0.yaml)|
|MobileNetV3_small_x1_25|70.7|6.9589|14.3995|13.0 M|[MobileNetV3_small_x1_25.yaml](../../paddlex/configs/image_classification/MobileNetV3_small_x1_25.yaml)|
|MobileNetV4_conv_large|83.4|12.5485|51.6453|125.2 M|[MobileNetV4_conv_large.yaml](../../paddlex/configs/image_classification/MobileNetV4_conv_large.yaml)|
|MobileNetV4_conv_medium|79.9|9.65509|26.6157|37.6 M|[MobileNetV4_conv_medium.yaml](../../paddlex/configs/image_classification/MobileNetV4_conv_medium.yaml)|
|MobileNetV4_conv_small|74.6|5.24172|11.0893|14.7 M|[MobileNetV4_conv_small.yaml](../../paddlex/configs/image_classification/MobileNetV4_conv_small.yaml)|
|MobileNetV4_hybrid_large|83.8|20.0726|213.769|145.1 M|[MobileNetV4_hybrid_large.yaml](../../paddlex/configs/image_classification/MobileNetV4_hybrid_large.yaml)|
|MobileNetV4_hybrid_medium|80.5|19.7543|62.2624|42.9 M|[MobileNetV4_hybrid_medium.yaml](../../paddlex/configs/image_classification/MobileNetV4_hybrid_medium.yaml)|
|PP-HGNet_base|85.0|14.2969|327.114|249.4 M|[PP-HGNet_base.yaml](../../paddlex/configs/image_classification/PP-HGNet_base.yaml)|
|PP-HGNet_small|81.51|5.50661|119.041|86.5 M|[PP-HGNet_small.yaml](../../paddlex/configs/image_classification/PP-HGNet_small.yaml)|
|PP-HGNet_tiny|79.83|5.22006|69.396|52.4 M|[PP-HGNet_tiny.yaml](../../paddlex/configs/image_classification/PP-HGNet_tiny.yaml)|
|PP-HGNetV2-B0|77.77|6.53694|23.352|21.4 M|[PP-HGNetV2-B0.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B0.yaml)|
|PP-HGNetV2-B1|79.18|6.56034|27.3099|22.6 M|[PP-HGNetV2-B1.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B1.yaml)|
|PP-HGNetV2-B2|81.74|9.60494|43.1219|39.9 M|[PP-HGNetV2-B2.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B2.yaml)|
|PP-HGNetV2-B3|82.98|11.0042|55.1367|57.9 M|[PP-HGNetV2-B3.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B3.yaml)|
|PP-HGNetV2-B4|83.57|9.66407|54.2462|70.4 M|[PP-HGNetV2-B4.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B4.yaml)|
|PP-HGNetV2-B5|84.75|15.7091|115.926|140.8 M|[PP-HGNetV2-B5.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B5.yaml)|
|PP-HGNetV2-B6|86.30|21.226|255.279|268.4 M|[PP-HGNetV2-B6.yaml](../../paddlex/configs/image_classification/PP-HGNetV2-B6.yaml)|
|PP-LCNet_x0_5|63.14|3.67722|6.66857|6.7 M|[PP-LCNet_x0_5.yaml](../../paddlex/configs/image_classification/PP-LCNet_x0_5.yaml)|
|PP-LCNet_x0_25|51.86|2.65341|5.81357|5.5 M|[PP-LCNet_x0_25.yaml](../../paddlex/configs/image_classification/PP-LCNet_x0_25.yaml)|
|PP-LCNet_x0_35|58.09|2.7212|6.28944|5.9 M|[PP-LCNet_x0_35.yaml](../../paddlex/configs/image_classification/PP-LCNet_x0_35.yaml)|
|PP-LCNet_x0_75|68.18|3.91032|8.06953|8.4 M|[PP-LCNet_x0_75.yaml](../../paddlex/configs/image_classification/PP-LCNet_x0_75.yaml)|
|PP-LCNet_x1_0|71.32|3.84845|9.23735|10.5 M|[PP-LCNet_x1_0.yaml](../../paddlex/configs/image_classification/PP-LCNet_x1_0.yaml)|
|PP-LCNet_x1_5|73.71|3.97666|12.3457|16.0 M|[PP-LCNet_x1_5.yaml](../../paddlex/configs/image_classification/PP-LCNet_x1_5.yaml)|
|PP-LCNet_x2_0|75.18|4.07556|16.2752|23.2 M|[PP-LCNet_x2_0.yaml](../../paddlex/configs/image_classification/PP-LCNet_x2_0.yaml)|
|PP-LCNet_x2_5|76.60|4.06028|21.5063|32.1 M|[PP-LCNet_x2_5.yaml](../../paddlex/configs/image_classification/PP-LCNet_x2_5.yaml)|
|PP-LCNetV2_base|77.05|5.23428|19.6005|23.7 M|[PP-LCNetV2_base.yaml](../../paddlex/configs/image_classification/PP-LCNetV2_base.yaml)|
|PP-LCNetV2_large |78.51|6.78335|30.4378|37.3 M|[PP-LCNetV2_large.yaml](../../paddlex/configs/image_classification/PP-LCNetV2_large.yaml)|
|PP-LCNetV2_small|73.97|3.89762|13.0273|14.6 M|[PP-LCNetV2_small.yaml](../../paddlex/configs/image_classification/PP-LCNetV2_small.yaml)|
|ResNet18_vd|72.3|3.53048|31.3014|41.5 M|[ResNet18_vd.yaml](../../paddlex/configs/image_classification/ResNet18_vd.yaml)|
|ResNet18|71.0|2.4868|27.4601|41.5 M|[ResNet18.yaml](../../paddlex/configs/image_classification/ResNet18.yaml)|
|ResNet34_vd|76.0|5.60675|56.0653|77.3 M|[ResNet34_vd.yaml](../../paddlex/configs/image_classification/ResNet34_vd.yaml)|
|ResNet34|74.6|4.16902|51.925|77.3 M|[ResNet34.yaml](../../paddlex/configs/image_classification/ResNet34.yaml)|
|ResNet50_vd|79.1|10.1885|68.446|90.8 M|[ResNet50_vd.yaml](../../paddlex/configs/image_classification/ResNet50_vd.yaml)|
|ResNet50|76.5|9.62383|64.8135|90.8 M|[ResNet50.yaml](../../paddlex/configs/image_classification/ResNet50.yaml)|
|ResNet101_vd|80.2|20.0563|124.85|158.4 M|[ResNet101_vd.yaml](../../paddlex/configs/image_classification/ResNet101_vd.yaml)|
|ResNet101|77.6|19.2297|121.006|158.7 M|[ResNet101.yaml](../../paddlex/configs/image_classification/ResNet101.yaml)|
|ResNet152_vd|80.6|29.6439|181.678|214.3 M|[ResNet152_vd.yaml](../../paddlex/configs/image_classification/ResNet152_vd.yaml)|
|ResNet152|78.3|30.0461|177.707|214.2 M|[ResNet152.yaml](../../paddlex/configs/image_classification/ResNet152.yaml)|
|ResNet200_vd|80.9|39.1628|235.185|266.0 M|[ResNet200_vd.yaml](../../paddlex/configs/image_classification/ResNet200_vd.yaml)|
|StarNet-S1|73.6|9.895|23.0465|11.2 M|[StarNet-S1.yaml](../../paddlex/configs/image_classification/StarNet-S1.yaml)|
|StarNet-S2|74.8|7.91279|21.9571|14.3 M|[StarNet-S2.yaml](../../paddlex/configs/image_classification/StarNet-S2.yaml)|
|StarNet-S3|77.0|10.7531|30.7656|22.2 M|[StarNet-S3.yaml](../../paddlex/configs/image_classification/StarNet-S3.yaml)|
|StarNet-S4|79.0|15.2868|43.2497|28.9 M|[StarNet-S4.yaml](../../paddlex/configs/image_classification/StarNet-S4.yaml)|
|SwinTransformer_base_patch4_window7_224|83.37|16.9848|383.83|310.5 M|[SwinTransformer_base_patch4_window7_224.yaml](../../paddlex/configs/image_classification/SwinTransformer_base_patch4_window7_224.yaml)|
|SwinTransformer_base_patch4_window12_384|84.17|37.2855|1178.63|311.4 M|[SwinTransformer_base_patch4_window12_384.yaml](../../paddlex/configs/image_classification/SwinTransformer_base_patch4_window12_384.yaml)|
|SwinTransformer_large_patch4_window7_224|86.19|27.5498|689.729|694.8 M|[SwinTransformer_large_patch4_window7_224.yaml](../../paddlex/configs/image_classification/SwinTransformer_large_patch4_window7_224.yaml)|
|SwinTransformer_large_patch4_window12_384|87.06|74.1768|2105.22|696.1 M|[SwinTransformer_large_patch4_window12_384.yaml](../../paddlex/configs/image_classification/SwinTransformer_large_patch4_window12_384.yaml)|
|SwinTransformer_small_patch4_window7_224|83.21|16.3982|285.56|175.6 M|[SwinTransformer_small_patch4_window7_224.yaml](../../paddlex/configs/image_classification/SwinTransformer_small_patch4_window7_224.yaml)|
|SwinTransformer_tiny_patch4_window7_224|81.10|8.54846|156.306|100.1 M|[SwinTransformer_tiny_patch4_window7_224.yaml](../../paddlex/configs/image_classification/SwinTransformer_tiny_patch4_window7_224.yaml)|

**注：以上精度指标为 **[ImageNet-1k](https://www.image-net.org/index.php)** 验证集 Top1 Acc。**

## [图像多标签分类模块](../module_usage/tutorials/cv_modules/ml_classification.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|CLIP_vit_base_patch16_448_ML|89.15|-|-|325.6 M|[CLIP_vit_base_patch16_448_ML.yaml](../../paddlex/configs/multilabel_classification/CLIP_vit_base_patch16_448_ML.yaml)|
|PP-HGNetV2-B0_ML|80.98|-|-|39.6 M|[PP-HGNetV2-B0_ML.yaml](../../paddlex/configs/multilabel_classification/PP-HGNetV2-B0_ML.yaml)|
|PP-HGNetV2-B4_ML|87.96|-|-|88.5 M|[PP-HGNetV2-B4_ML.yaml](../../paddlex/configs/multilabel_classification/PP-HGNetV2-B4_ML.yaml)|
|PP-HGNetV2-B6_ML|91.25|-|-|286.5 M|[PP-HGNetV2-B6_ML.yaml](../../paddlex/configs/multilabel_classification/PP-HGNetV2-B6_ML.yaml)|
|PP-LCNet_x1_0_ML|77.96|-|-|29.4 M|[PP-LCNet_x1_0_ML.yaml](../../paddlex/configs/multilabel_classification/PP-LCNet_x1_0_ML.yaml)|
|ResNet50_ML|83.50|-|-|108.9 M|[ResNet50_ML.yaml](../../paddlex/configs/multilabel_classification/ResNet50_ML.yaml)|

**注：以上精度指标为 [COCO2017](https://cocodataset.org/#home) 的多标签分类任务mAP。**

## [行人属性模块](../module_usage/tutorials/cv_modules/pedestrian_attribute_recognition.md)
|模型名称|mA（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-LCNet_x1_0_pedestrian_attribute|92.2|3.84845|9.23735|6.7 M  |[PP-LCNet_x1_0_pedestrian_attribute.yaml](../../paddlex/configs/pedestrian_attribute/PP-LCNet_x1_0_pedestrian_attribute.yaml)|

**注：以上精度指标为 PaddleX 内部自建数据集mA。**

## [车辆属性模块](../module_usage/tutorials/cv_modules/vehicle_attribute_recognition.md)
|模型名称|mA（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-LCNet_x1_0_vehicle_attribute|91.7|3.84845|9.23735|6.7 M|[PP-LCNet_x1_0_vehicle_attribute.yaml](../../paddlex/configs/vehicle_attribute/PP-LCNet_x1_0_vehicle_attribute.yaml)|

**注：以上精度指标为 VeRi 数据集 mA。**

## [图像特征模块](../module_usage/tutorials/cv_modules/image_feature.md)
|模型名称|recall@1（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-ShiTuV2_rec|84.2|5.23428|19.6005|16.3 M|[PP-ShiTuV2_rec.yaml](../../paddlex/configs/general_recognition/PP-ShiTuV2_rec.yaml)|
|PP-ShiTuV2_rec_CLIP_vit_base|88.69|13.1957|285.493|306.6 M|[PP-ShiTuV2_rec_CLIP_vit_base.yaml](../../paddlex/configs/general_recognition/PP-ShiTuV2_rec_CLIP_vit_base.yaml)|
|PP-ShiTuV2_rec_CLIP_vit_large|91.03|51.1284|1131.28|1.05 G|[PP-ShiTuV2_rec_CLIP_vit_large.yaml](../../paddlex/configs/general_recognition/PP-ShiTuV2_rec_CLIP_vit_large.yaml)|

**注：以上精度指标为 AliProducts recall@1。**

## [文档方向分类模块](../module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)
|模型名称|Top-1 Acc（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-LCNet_x1_0_doc_ori|99.26|3.84845|9.23735|7.1 M|[PP-LCNet_x1_0_doc_ori.yaml](../../paddlex/configs/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml)|

**注：以上精度指标为 PaddleX 内部自建数据集 Top-1 Acc 。**

## [主体检测模块](../module_usage/tutorials/cv_modules/mainbody_detection.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-ShiTuV2_det|41.5|33.7426|537.003|27.6 M|[PP-ShiTuV2_det.yaml](../../paddlex/configs/mainbody_detection/PP-ShiTuV2_det.yaml)|

**注：以上精度指标为 [PaddleClas主体检测数据集](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.5/docs/zh_CN/training/PP-ShiTu/mainbody_detection.md) mAP(0.5:0.95)。**

## [目标检测模块](../module_usage/tutorials/cv_modules/object_detection.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|Cascade-FasterRCNN-ResNet50-FPN|41.1|-|-|245.4 M|[Cascade-FasterRCNN-ResNet50-FPN.yaml](../../paddlex/configs/object_detection/Cascade-FasterRCNN-ResNet50-FPN.yaml)|
|Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN|45.0|-|-|246.2 M|[Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml](../../paddlex/configs/object_detection/Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml)|
|CenterNet-DLA-34|37.6|-|-|75.4 M|[CenterNet-DLA-34.yaml](../../paddlex/configs/object_detection/CenterNet-DLA-34.yaml)|
|CenterNet-ResNet50|38.9|-|-|319.7 M|[CenterNet-ResNet50.yaml](../../paddlex/configs/object_detection/CenterNet-ResNet50.yaml)|
|DETR-R50|42.3|59.2132|5334.52|159.3 M|[DETR-R50.yaml](../../paddlex/configs/object_detection/DETR-R50.yaml)|
|FasterRCNN-ResNet34-FPN|37.8|-|-|137.5 M|[FasterRCNN-ResNet34-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet34-FPN.yaml)|
|FasterRCNN-ResNet50-FPN|38.4|-|-|148.1 M|[FasterRCNN-ResNet50-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet50-FPN.yaml)|
|FasterRCNN-ResNet50-vd-FPN|39.5|-|-|148.1 M|[FasterRCNN-ResNet50-vd-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet50-vd-FPN.yaml)|
|FasterRCNN-ResNet50-vd-SSLDv2-FPN|41.4|-|-|148.1 M|[FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet50-vd-SSLDv2-FPN.yaml)|
|FasterRCNN-ResNet50|36.7|-|-|120.2 M|[FasterRCNN-ResNet50.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet50.yaml)|
|FasterRCNN-ResNet101-FPN|41.4|-|-|216.3 M|[FasterRCNN-ResNet101-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet101-FPN.yaml)|
|FasterRCNN-ResNet101|39.0|-|-|188.1 M|[FasterRCNN-ResNet101.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNet101.yaml)|
|FasterRCNN-ResNeXt101-vd-FPN|43.4|-|-|360.6 M|[FasterRCNN-ResNeXt101-vd-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-ResNeXt101-vd-FPN.yaml)|
|FasterRCNN-Swin-Tiny-FPN|42.6|-|-|159.8 M|[FasterRCNN-Swin-Tiny-FPN.yaml](../../paddlex/configs/object_detection/FasterRCNN-Swin-Tiny-FPN.yaml)|
|FCOS-ResNet50|39.6|103.367|3424.91|124.2 M|[FCOS-ResNet50.yaml](../../paddlex/configs/object_detection/FCOS-ResNet50.yaml)|
|PicoDet-L|42.6|16.6715|169.904|20.9 M|[PicoDet-L.yaml](../../paddlex/configs/object_detection/PicoDet-L.yaml)|
|PicoDet-M|37.5|16.2311|71.7257|16.8 M|[PicoDet-M.yaml](../../paddlex/configs/object_detection/PicoDet-M.yaml)|
|PicoDet-S|29.1|14.097|37.6563|4.4 M |[PicoDet-S.yaml](../../paddlex/configs/object_detection/PicoDet-S.yaml)|
|PicoDet-XS|26.2|13.8102|48.3139|5.7M |[PicoDet-XS.yaml](../../paddlex/configs/object_detection/PicoDet-XS.yaml)|
|PP-YOLOE_plus-L|52.9|33.5644|814.825|185.3 M|[PP-YOLOE_plus-L.yaml](../../paddlex/configs/object_detection/PP-YOLOE_plus-L.yaml)|
|PP-YOLOE_plus-M|49.8|19.843|449.261|83.2 M|[PP-YOLOE_plus-M.yaml](../../paddlex/configs/object_detection/PP-YOLOE_plus-M.yaml)|
|PP-YOLOE_plus-S|43.7|16.8884|223.059|28.3 M|[PP-YOLOE_plus-S.yaml](../../paddlex/configs/object_detection/PP-YOLOE_plus-S.yaml)|
|PP-YOLOE_plus-X|54.7|57.8995|1439.93|349.4 M|[PP-YOLOE_plus-X.yaml](../../paddlex/configs/object_detection/PP-YOLOE_plus-X.yaml)|
|RT-DETR-H|56.3|114.814|3933.39|435.8 M|[RT-DETR-H.yaml](../../paddlex/configs/object_detection/RT-DETR-H.yaml)|
|RT-DETR-L|53.0|34.5252|1454.27|113.7 M|[RT-DETR-L.yaml](../../paddlex/configs/object_detection/RT-DETR-L.yaml)|
|RT-DETR-R18|46.5|19.89|784.824|70.7 M|[RT-DETR-R18.yaml](../../paddlex/configs/object_detection/RT-DETR-R18.yaml)|
|RT-DETR-R50|53.1|41.9327|1625.95|149.1 M|[RT-DETR-R50.yaml](../../paddlex/configs/object_detection/RT-DETR-R50.yaml)|
|RT-DETR-X|54.8|61.8042|2246.64|232.9 M|[RT-DETR-X.yaml](../../paddlex/configs/object_detection/RT-DETR-X.yaml)|
|YOLOv3-DarkNet53|39.1|40.1055|883.041|219.7 M|[YOLOv3-DarkNet53.yaml](../../paddlex/configs/object_detection/YOLOv3-DarkNet53.yaml)|
|YOLOv3-MobileNetV3|31.4|18.6692|267.214|83.8 M|[YOLOv3-MobileNetV3.yaml](../../paddlex/configs/object_detection/YOLOv3-MobileNetV3.yaml)|
|YOLOv3-ResNet50_vd_DCN|40.6|31.6276|856.047|163.0 M|[YOLOv3-ResNet50_vd_DCN.yaml](../../paddlex/configs/object_detection/YOLOv3-ResNet50_vd_DCN.yaml)|
|YOLOX-L|50.1|185.691|1250.58|192.5 M|[YOLOX-L.yaml](../../paddlex/configs/object_detection/YOLOX-L.yaml)|
|YOLOX-M|46.9|123.324|688.071|90.0 M|[YOLOX-M.yaml](../../paddlex/configs/object_detection/YOLOX-M.yaml)|
|YOLOX-N|26.1|79.1665|155.59|3.4M|[YOLOX-N.yaml](../../paddlex/configs/object_detection/YOLOX-N.yaml)|
|YOLOX-S|40.4|184.828|474.446|32.0 M|[YOLOX-S.yaml](../../paddlex/configs/object_detection/YOLOX-S.yaml)|
|YOLOX-T|32.9|102.748|212.52|18.1 M|[YOLOX-T.yaml](../../paddlex/configs/object_detection/YOLOX-T.yaml)|
|YOLOX-X|51.8|227.361|2067.84|351.5 M|[YOLOX-X.yaml](../../paddlex/configs/object_detection/YOLOX-X.yaml)|

**注：以上精度指标为 **[COCO2017](https://cocodataset.org/#home)** 验证集 mAP(0.5:0.95)。**

## [小目标检测模块](../module_usage/tutorials/cv_modules/small_object_detection.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-YOLOE_plus_SOD-S|25.1|65.4608|324.37|77.3 M|[PP-YOLOE_plus_SOD-S.yaml](../../paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-S.yaml)|
|PP-YOLOE_plus_SOD-L|31.9|57.1448|1006.98|325.0 M|[PP-YOLOE_plus_SOD-L.yaml](../../paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-L.yaml)|
|PP-YOLOE_plus_SOD-largesize-L|42.7|458.521|11172.7|340.5 M|[PP-YOLOE_plus_SOD-largesize-L.yaml](../../paddlex/configs/small_object_detection/PP-YOLOE_plus_SOD-largesize-L.yaml)|

**注：以上精度指标为 **[VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset)** 验证集 mAP(0.5:0.95)。**

## [行人检测模块](../module_usage/tutorials/cv_modules/human_detection.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-YOLOE-L_human|48.0|32.7754|777.691|196.1 M|[PP-YOLOE-L_human.yaml](../../paddlex/configs/human_detection/PP-YOLOE-L_human.yaml)|
|PP-YOLOE-S_human|42.5|15.0118|179.317|28.8 M|[PP-YOLOE-S_human.yaml](../../paddlex/configs/human_detection/PP-YOLOE-S_human.yaml)|

**注：以上精度指标为 **[CrowdHuman](https://bj.bcebos.com/v1/paddledet/data/crowdhuman.zip)** 验证集 mAP(0.5:0.95)。**

## [车辆检测模块](../module_usage/tutorials/cv_modules/vehicle_detection.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-YOLOE-L_vehicle|63.9|32.5619|775.633|196.1 M|[PP-YOLOE-L_vehicle.yaml](../../paddlex/configs/vehicle_detection/PP-YOLOE-L_vehicle.yaml)|
|PP-YOLOE-S_vehicle|61.3|15.3787|178.441|28.8 M|[PP-YOLOE-S_vehicle.yaml](../../paddlex/configs/vehicle_detection/PP-YOLOE-S_vehicle.yaml)|

**注：以上精度指标为 **[PPVehicle](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppvehicle)** 验证集 mAP(0.5:0.95)。**

## [人脸检测模块](../module_usage/tutorials/cv_modules/face_detection.md)
|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PicoDet_LCNet_x2_5_face|35.8|33.7426|537.003|27.7 M|[PicoDet_LCNet_x2_5_face.yaml](../../paddlex/configs/face_detection/PicoDet_LCNet_x2_5_face.yaml)|

**注：以上精度指标为 **[wider_face](https://github.com/PaddlePaddle/PaddleDetection/tree/develop/configs/ppvehicle)** 评估集 mAP(0.5:0.95)。**

## [异常检测模块](../module_usage/tutorials/cv_modules/anomaly_detection.md)
|模型名称|Avg（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|STFPM|96.2|-|-|21.5 M|[STFPM.yaml](../../paddlex/configs/anomaly_detection/STFPM.yaml)|

**注：以上精度指标为 **[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)** 验证集 平均异常分数。**

## [语义分割模块](../module_usage/tutorials/cv_modules/semantic_segmentation.md)
|模型名称|mloU（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|Deeplabv3_Plus-R50 |80.36|61.0531|1513.58|94.9 M|[Deeplabv3_Plus-R50.yaml](../../paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R50.yaml)|
|Deeplabv3_Plus-R101|81.10|100.026|2460.71|162.5 M|[Deeplabv3_Plus-R101.yaml](../../paddlex/configs/semantic_segmentation/Deeplabv3_Plus-R101.yaml)|
|Deeplabv3-R50|79.90|82.2631|1735.83|138.3 M|[Deeplabv3-R50.yaml](../../paddlex/configs/semantic_segmentation/Deeplabv3-R50.yaml)|
|Deeplabv3-R101|80.85|121.492|2685.51|205.9 M|[Deeplabv3-R101.yaml](../../paddlex/configs/semantic_segmentation/Deeplabv3-R101.yaml)|
|OCRNet_HRNet-W18|80.67|48.2335|906.385|43.1 M|[OCRNet_HRNet-W18.yaml](../../paddlex/configs/semantic_segmentation/OCRNet_HRNet-W18.yaml)|
|OCRNet_HRNet-W48|82.15|78.9976|2226.95|249.8 M|[OCRNet_HRNet-W48.yaml](../../paddlex/configs/semantic_segmentation/OCRNet_HRNet-W48.yaml)|
|PP-LiteSeg-T|73.10|7.6827|138.683|28.5 M|[PP-LiteSeg-T.yaml](../../paddlex/configs/semantic_segmentation/PP-LiteSeg-T.yaml)|
|PP-LiteSeg-B|75.25|-|-|47.0 M|[PP-LiteSeg-B.yaml](../../paddlex/configs/semantic_segmentation/PP-LiteSeg-B.yaml)|
|SegFormer-B0 (slice)|76.73|11.1946|268.929|13.2 M|[SegFormer-B0.yaml](../../paddlex/configs/semantic_segmentation/SegFormer-B0.yaml)|
|SegFormer-B1 (slice)|78.35|17.9998|403.393|48.5 M|[SegFormer-B1.yaml](../../paddlex/configs/semantic_segmentation/SegFormer-B1.yaml)|
|SegFormer-B2 (slice)|81.60|48.0371|1248.52|96.9 M|[SegFormer-B2.yaml](../../paddlex/configs/semantic_segmentation/SegFormer-B2.yaml)|
|SegFormer-B3 (slice)|82.47|64.341|1666.35|167.3 M|[SegFormer-B3.yaml](../../paddlex/configs/semantic_segmentation/SegFormer-B3.yaml)|
|SegFormer-B4 (slice)|82.38|82.4336|1995.42|226.7 M|[SegFormer-B4.yaml](../../paddlex/configs/semantic_segmentation/SegFormer-B4.yaml)|
|SegFormer-B5 (slice)|82.58|97.3717|2420.19|229.7 M|[SegFormer-B5.yaml](../../paddlex/configs/semantic_segmentation/SegFormer-B5.yaml)|

**注：以上精度指标为 **[Cityscapes](https://www.cityscapes-dataset.com/)** 数据集 mloU。**

|模型名称|mloU（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|SeaFormer_base(slice)|40.92|24.4073|397.574|30.8 M|[SeaFormer_base.yaml](../../paddlex/configs/semantic_segmentation/SeaFormer_base.yaml)|
|SeaFormer_large (slice)|43.66|27.8123|550.464|49.8 M|[SeaFormer_large.yaml](../../paddlex/configs/semantic_segmentation/SeaFormer_large.yaml)|
|SeaFormer_small (slice)|38.73|19.2295|358.343|14.3 M|[SeaFormer_small.yaml](../../paddlex/configs/semantic_segmentation/SeaFormer_small.yaml)|
|SeaFormer_tiny (slice)|34.58|13.9496|330.132|6.1M |[SeaFormer_tiny.yaml](../../paddlex/configs/semantic_segmentation/SeaFormer_tiny.yaml)|

**注：以上精度指标为 **[ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/)** 数据集, slice 表示对输入图像进行了切图操作。**

## [实例分割模块](../module_usage/tutorials/cv_modules/instance_segmentation.md)
|模型名称|Mask AP|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|Mask-RT-DETR-H|50.6|132.693|4896.17|449.9 M|[Mask-RT-DETR-H.yaml](../../paddlex/configs/instance_segmentation/Mask-RT-DETR-H.yaml)|
|Mask-RT-DETR-L|45.7|46.5059|2575.92|113.6 M|[Mask-RT-DETR-L.yaml](../../paddlex/configs/instance_segmentation/Mask-RT-DETR-L.yaml)|
|Mask-RT-DETR-M|42.7|36.8329|-|66.6 M|[Mask-RT-DETR-M.yaml](../../paddlex/configs/instance_segmentation/Mask-RT-DETR-M.yaml)|
|Mask-RT-DETR-S|41.0|33.5007|-|51.8 M|[Mask-RT-DETR-S.yaml](../../paddlex/configs/instance_segmentation/Mask-RT-DETR-S.yaml)|
|Mask-RT-DETR-X|47.5|75.755|3358.04|237.5 M|[Mask-RT-DETR-X.yaml](../../paddlex/configs/instance_segmentation/Mask-RT-DETR-X.yaml)|
|Cascade-MaskRCNN-ResNet50-FPN|36.3|-|-|254.8 M|[Cascade-MaskRCNN-ResNet50-FPN.yaml](../../paddlex/configs/instance_segmentation/Cascade-MaskRCNN-ResNet50-FPN.yaml)|
|Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN|39.1|-|-|254.7 M|[Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml](../../paddlex/configs/instance_segmentation/Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN.yaml)|
|MaskRCNN-ResNet50-FPN|35.6|-|-|157.5 M|[MaskRCNN-ResNet50-FPN.yaml](../../paddlex/configs/instance_segmentation/MaskRCNN-ResNet50-FPN.yaml)|
|MaskRCNN-ResNet50-vd-FPN|36.4|-|-|157.5 M|[MaskRCNN-ResNet50-vd-FPN.yaml](../../paddlex/configs/instance_segmentation/MaskRCNN-ResNet50-vd-FPN.yaml)|
|MaskRCNN-ResNet50|32.8|-|-|127.8 M|[MaskRCNN-ResNet50.yaml](../../paddlex/configs/instance_segmentation/MaskRCNN-ResNet50.yaml)|
|MaskRCNN-ResNet101-FPN|36.6|-|-|225.4 M|[MaskRCNN-ResNet101-FPN.yaml](../../paddlex/configs/instance_segmentation/MaskRCNN-ResNet101-FPN.yaml)|
|MaskRCNN-ResNet101-vd-FPN|38.1|-|-|225.1 M|[MaskRCNN-ResNet101-vd-FPN.yaml](../../paddlex/configs/instance_segmentation/MaskRCNN-ResNet101-vd-FPN.yaml)|
|MaskRCNN-ResNeXt101-vd-FPN|39.5|-|-|370.0 M|[MaskRCNN-ResNeXt101-vd-FPN.yaml](../../paddlex/configs/instance_segmentation/MaskRCNN-ResNeXt101-vd-FPN.yaml)|
|PP-YOLOE_seg-S|32.5|-|-|31.5 M|[PP-YOLOE_seg-S.yaml](../../paddlex/configs/instance_segmentation/PP-YOLOE_seg-S.yaml)|
|SOLOv2| 35.5|-|-|179.1 M|[SOLOv2.yaml](../../paddlex/configs/instance_segmentation/SOLOv2.yaml)

**注：以上精度指标为 **[COCO2017](https://cocodataset.org/#home)** 验证集 Mask AP(0.5:0.95)。**

## [文本检测模块](../module_usage/tutorials/ocr_modules/text_detection.md)
|模型名称|检测Hmean（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-OCRv4_mobile_det |77.79|10.6923|120.177|4.2 M|[PP-OCRv4_mobile_det.yaml](../../paddlex/configs/text_detection/PP-OCRv4_mobile_det.yaml)|
|PP-OCRv4_server_det |82.69|83.3501|2434.01|100.1M|[PP-OCRv4_server_det.yaml](../../paddlex/configs/text_detection/PP-OCRv4_server_det.yaml)|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。**

## [印章文本检测模块](../module_usage/tutorials/ocr_modules/seal_text_detection.md)
|模型名称|检测Hmean（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-OCRv4_mobile_seal_det|96.47|10.5878|131.813|4.7M |[PP-OCRv4_mobile_seal_det.yaml](../../paddlex/configs/text_detection_seal/PP-OCRv4_mobile_seal_det.yaml)|
|PP-OCRv4_server_seal_det|98.21|84.341|2425.06|108.3 M|[PP-OCRv4_server_seal_det.yaml](../../paddlex/configs/text_detection_seal/PP-OCRv4_server_seal_det.yaml)|

**注：以上精度指标的评估集是 PaddleX 自建的印章数据集，包含500印章图像。**

## [文本识别模块](../module_usage/tutorials/ocr_modules/text_recognition.md)
|模型名称|识别Avg Accuracy(%)|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PP-OCRv4_mobile_rec |78.20|7.95018|46.7868|10.6 M|[PP-OCRv4_mobile_rec.yaml](../../paddlex/configs/text_recognition/PP-OCRv4_mobile_rec.yaml)|
|PP-OCRv4_server_rec |79.20|7.19439|140.179|71.2 M|[PP-OCRv4_server_rec.yaml](../../paddlex/configs/text_recognition/PP-OCRv4_server_rec.yaml)|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。**

|模型名称|识别Avg Accuracy(%)|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|ch_SVTRv2_rec|68.81|8.36801|165.706|73.9 M|[ch_SVTRv2_rec.yaml](../../paddlex/configs/text_recognition/ch_SVTRv2_rec.yaml)|

**注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)A榜。**

|模型名称|识别Avg Accuracy(%)|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|ch_RepSVTR_rec|65.07|10.5047|51.5647|22.1 M|[ch_RepSVTR_rec.yaml](../../paddlex/configs/text_recognition/ch_RepSVTR_rec.yaml)|

**注：以上精度指标的评估集是 [PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)B榜。**

## [公式识别模块](../module_usage/tutorials/ocr_modules/formula_recognition.md)
|模型名称|BLEU score|normed edit distance|ExpRate （%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|-|-|
|LaTeX_OCR_rec|0.8821|0.0823|40.01|-|-|89.7 M|[LaTeX_OCR_rec.yaml](../../paddlex/configs/formula_recognition/LaTeX_OCR_rec.yaml)|

**注：以上精度指标测量自 [LaTeX-OCR公式识别测试集](https://drive.google.com/drive/folders/13CA4vAmOmD_I_dSbvLp-Lf0s6KiaNfuO)。**

## [表格结构识别模块](../module_usage/tutorials/ocr_modules/table_structure_recognition.md)
|模型名称|精度（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|SLANet|59.52|522.536|1845.37|6.9 M |[SLANet.yaml](../../paddlex/configs/table_recognition/SLANet.yaml)|
|SLANet_plus|63.69|522.536|1845.37|6.9 M |[SLANet_plus.yaml](../../paddlex/configs/table_recognition/SLANet_plus.yaml)|

**注：以上精度指标测量自 ****PaddleX内部自建英文表格识别数据集****。**

## [图像矫正模块](../module_usage/tutorials/ocr_modules/text_image_unwarping.md)
|模型名称|MS-SSIM （%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|UVDoc|54.40|-|-|30.3 M|[UVDoc.yaml](../../paddlex/configs/image_unwarping/UVDoc.yaml)|

**注：以上精度指标测量自 ****PaddleX自建的图像矫正数据集****。**

## [版面区域检测模块](../module_usage/tutorials/ocr_modules/layout_detection.md)
|模型名称|mAP@(0.50:0.95)（%）|GPU推理耗时（ms）|CPU推理耗时（ms）|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|PicoDet_layout_1x|86.8|13.036|91.2634|7.4 M |[PicoDet_layout_1x.yaml](../../paddlex/configs/structure_analysis/PicoDet_layout_1x.yaml)|
|PicoDet-S_layout_3cls|87.1|?|?|4.8 M|[PicoDet-S_layout_3cls.yaml](../../paddlex/configs/structure_analysis/PicoDet-S_layout_3cls.yaml)|
|PicoDet-S_layout_17cls|70.3|?|?|4.8 M|[PicoDet-S_layout_17cls.yaml](../../paddlex/configs/structure_analysis/PicoDet-S_layout_17cls.yaml)|
|PicoDet-L_layout_3cls|89.3|15.7425|159.771|22.6 M|[PicoDet-L_layout_3cls.yaml](../../paddlex/configs/structure_analysis/PicoDet-L_layout_3cls.yaml)|
|PicoDet-L_layout_17cls|79.9|?|?|22.6 M|[PicoDet-L_layout_17cls.yaml](../../paddlex/configs/structure_analysis/PicoDet-L_layout_17cls.yaml)|
|RT-DETR-H_layout_3cls|95.9|114.644|3832.62|470.1 M|[RT-DETR-H_layout_3cls.yaml](../../paddlex/configs/structure_analysis/RT-DETR-H_layout_3cls.yaml)|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2 M|[RT-DETR-H_layout_17cls.yaml](../../paddlex/configs/structure_analysis/RT-DETR-H_layout_17cls.yaml)|

**注：以上精度指标的评估集是 ****PaddleX 自建的版面区域检测数据集****，包含 1w 张图片。**

## [时序预测模块](../module_usage/tutorials/time_series_modules/time_series_forecasting.md)
|模型名称|mse|mae|模型存储大小|yaml 文件|
|-|-|-|-|-|
|DLinear|0.382|0.394|72 K|[DLinear.yaml](../../paddlex/configs/ts_forecast/DLinear.yaml)|
|NLinear|0.386|0.392|40 K |[NLinear.yaml](../../paddlex/configs/ts_forecast/NLinear.yaml)|
|Nonstationary|0.600|0.515|55.5 M|[Nonstationary.yaml](../../paddlex/configs/ts_forecast/Nonstationary.yaml)|
|PatchTST|0.385|0.397|2.0 M |[PatchTST.yaml](../../paddlex/configs/ts_forecast/PatchTST.yaml)|
|RLinear|0.384|0.392|40 K|[RLinear.yaml](../../paddlex/configs/ts_forecast/RLinear.yaml)|
|TiDE|0.405|0.412|31.7 M|[TiDE.yaml](../../paddlex/configs/ts_forecast/TiDE.yaml)|
|TimesNet|0.417|0.431|4.9 M|[TimesNet.yaml](../../paddlex/configs/ts_forecast/TimesNet.yaml)|

**注：以上精度指标测量自 **[ETTH1](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/Etth1.tar)** 数据集 ****（在测试集test.csv上的评测结果）****。**

## [时序异常检测模块](../module_usage/tutorials/time_series_modules/time_series_anomaly_detection.md)
|模型名称|precison|recall|f1_score|模型存储大小|yaml 文件|
|-|-|-|-|-|-|
|AutoEncoder_ad|99.36|84.36|91.25|52 K |[AutoEncoder_ad.yaml](../../paddlex/configs/ts_anomaly_detection/AutoEncoder_ad.yaml)|
|DLinear_ad|98.98|93.96|96.41|112 K|[DLinear_ad.yaml](../../paddlex/configs/ts_anomaly_detection/DLinear_ad.yaml)|
|Nonstationary_ad|98.55|88.95|93.51|1.8 M |[Nonstationary_ad.yaml](../../paddlex/configs/ts_anomaly_detection/Nonstationary_ad.yaml)|
|PatchTST_ad|98.78|90.70|94.57|320 K |[PatchTST_ad.yaml](../../paddlex/configs/ts_anomaly_detection/PatchTST_ad.yaml)|
|TimesNet_ad|98.37|94.80|96.56|1.3 M |[TimesNet_ad.yaml](../../paddlex/configs/ts_anomaly_detection/TimesNet_ad.yaml)|

**注：以上精度指标测量自 **[PSM](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ts_anomaly_examples.tar)** 数据集。**

## [时序分类模块](../module_usage/tutorials/time_series_modules/time_series_classification.md)
|模型名称|acc(%)|模型存储大小|yaml 文件|
|-|-|-|-|
|TimesNet_cls|87.5|792 K|[TimesNet_cls.yaml](../../paddlex/configs/ts_classification/TimesNet_cls.yaml)|

**注：以上精度指标测量自 [UWaveGestureLibrary](https://paddlets.bj.bcebos.com/classification/UWaveGestureLibrary_TEST.csv)数据集。**

>**注：以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**
