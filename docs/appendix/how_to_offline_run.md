# 无联网模型训练

PaddleX在模型训练时，存在以下两种情况需要进行联网下载
> 1.训练模型时，用户没有配置自定义的预训练模型权重`pretrain_weights`，此时PaddleX会自动联网下载在标准数据集上的预训练模型；
> 2.模型裁剪训练时，用户没有配置自定义的参数敏感度信息文件`sensitivities_file`，并将`sensitivities_file`配置成了'DEFAULT'字符串，此时PaddleX会自动联网下载模型在标准数据集上计算得到的参数敏感度信息文件。


## 如何在没联网的情况下进行模型训练
> 在训练模型时，不管是正常训练还是裁剪训练，用户可以提前准备好预训练权重或参数敏感度信息文档，只需自定义`pretrain_weights`或`sensitivities_file`， 将其设为本地的路径即可。


## 预训练模型下载地址
> 以下模型均为分类模型权重（UNet除外），用户在训练模型时，需要**根据分类模型的种类或backbone的种类**，选择对应的模型权重进行下载(目标检测在使用ResNet50作为Backbone时，使用下面表格中的ResNet50_cos作为预训练模型)

| 模型(点击下载) | 数据集 |
| :------------|:------|
| [ResNet18](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar) | ImageNet |
| [ResNet34](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar) | ImageNet |
| [ResNet50](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar) | ImageNet |
| [ResNet101](http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar) | ImageNet |
| [ResNet50_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar) | ImageNet |
| [ResNet101_vd](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar) | ImageNet |
| [MobileNetV1](http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar) | ImageNet |
| [MobileNetV2_x1.0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar) | ImageNet |
| [MobileNetV2_x0.5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar) | ImageNet |
| [MobileNetV2_x2.0](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar) | ImageNet |
| [MobileNetV2_x0.25](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar) | ImageNet |
| [MobileNetV2_x1.5](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar) | ImageNet |
| [MobileNetV3_small](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar) | ImageNet |
| [MobileNetV3_large](https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar) | ImageNet |
| [DarkNet53](https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar) | ImageNet |
| [DenseNet121](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar) | ImageNet |
| [DenseNet161](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar) | ImageNet |
| [DenseNet201](https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar) | ImageNet |
| [ResNet50_cos](https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar) | ImageNet |
| [Xception41](https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar) | ImageNet |
| [Xception65](https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar) | ImageNet |
| [ShuffleNetV2](https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar) | ImageNet |
| [UNet](https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz) | MSCOCO |
