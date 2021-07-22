# 无联网模型训练

PaddleX在模型训练时，存在以下两种情况需要进行联网下载
> 1.训练模型时，用户没有配置自定义的预训练模型权重`pretrain_weights`，此时PaddleX会自动联网下载在标准数据集上的预训练模型；
> 2.模型裁剪训练时，用户没有配置自定义的参数敏感度信息文件`sensitivities_file`，并将`sensitivities_file`配置成了'DEFAULT'字符串，此时PaddleX会自动联网下载模型在标准数据集上计算得到的参数敏感度信息文件。

## PaddleX Python API离线训练
> 通过如下代码先下载好PaddleX的所有预训练模型，下载完共约7.5G  
```
from paddlex.cv.models.utils.pretrain_weights import image_pretrain
from paddlex.cv.models.utils.pretrain_weights import coco_pretrain
from paddlex.cv.models.utils.pretrain_weights import cityscapes_pretrain
import paddlehub as hub

save_dir = '/home/work/paddlex_pretrain'
for name, url in image_pretrain.items():
    hub.download(name, save_dir)
for name, url in coco_pretrain.items():
    hub.download(name, save_dir)
for name, url in cityscapes_pretrain.items():
    hub.download(name, save_dir)
```

用户在可联网的机器上，执行如上代码，所有的预训练模型将会下载至指定的`save_dir`（代码示例中为`/home/work/paddlex_pretrain`），之后在通过Python代码使用PaddleX训练代码时，只需要在import paddlex的同时，配置如下参数，模型在训练时便会优先在此目录下寻找已经下载好的预训练模型。
```
import paddlex as pdx
pdx.pretrain_dir = '/home/work/paddlex_pretrain'
```

## PaddleX GUI离线训练
> PaddleX GUI在打开后，需要用户设定工作空间，假设当前用户设定的工作空间为`D:\PaddleX_Workspace`，为了离线训练，用户需手动下载如下所有文件（下载后无需再做解压操作)至`D:\PaddleX_Workspace\pretrain`目录，之后在训练模型时，便不再需要联网  
```
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet34_pretrained.tar
http://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_pretrained.tar
http://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet101_vd_ssld_pretrained.tar
http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_5_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x2_0_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x0_25_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV2_x1_5_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_ssld_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x1_0_ssld_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/DarkNet53_ImageNet1k_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet121_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet161_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/DenseNet201_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_cos_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/Xception41_deeplab_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/Xception65_deeplab_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/ShuffleNetV2_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W18_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W30_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W32_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W40_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W44_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W48_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W60_C_pretrained.tar
https://paddle-imagenet-models-name.bj.bcebos.com/HRNet_W64_C_pretrained.tar
http://paddle-imagenet-models-name.bj.bcebos.com/AlexNet_pretrained.tar
https://paddlemodels.bj.bcebos.com/object_detection/yolov3_darknet.tar
https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
https://bj.bcebos.com/paddlex/models/yolov3_mobilenet_v3.tar
https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34.tar
https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r50vd_dcn.tar
https://bj.bcebos.com/paddlex/pretrained_weights/faster_rcnn_r18_fpn_1x.tar
https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_fpn_2x.tar
https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r50_vd_fpn_2x.tar
https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_fpn_2x.tar
https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_r101_vd_fpn_2x.tar
https://paddlemodels.bj.bcebos.com/object_detection/faster_rcnn_hrnetv2p_w18_2x.tar
https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_r18_fpn_1x.tar
https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_fpn_2x.tar
https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r50_vd_fpn_2x.tar
https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_fpn_1x.tar
https://paddlemodels.bj.bcebos.com/object_detection/mask_rcnn_r101_vd_fpn_1x.tar
https://bj.bcebos.com/paddlex/pretrained_weights/mask_rcnn_hrnetv2p_w18_2x.tar
https://paddleseg.bj.bcebos.com/models/unet_coco_v3.tgz
https://bj.bcebos.com/v1/paddleseg/deeplab_mobilenet_x1_0_coco.tgz
https://paddleseg.bj.bcebos.com/models/xception65_coco.tgz
https://paddlemodels.bj.bcebos.com/object_detection/ppyolo_2x.pdparams
https://paddleseg.bj.bcebos.com/models/deeplabv3p_mobilenetv3_large_cityscapes.tar.gz
https://paddleseg.bj.bcebos.com/models/mobilenet_cityscapes.tgz
https://paddleseg.bj.bcebos.com/models/xception65_bn_cityscapes.tgz
https://paddleseg.bj.bcebos.com/models/hrnet_w18_bn_cityscapes.tgz
https://paddleseg.bj.bcebos.com/models/fast_scnn_cityscape.tar
```
