# 使用教程——训练模型

本目录下整理了使用PaddleX训练模型的示例代码，代码中均提供了示例数据的自动下载，并均使用单张GPU卡进行训练。

|代码 | 模型任务 | 数据 |
|------|--------|---------|
|classification/mobilenetv2.py | 图像分类MobileNetV2 | 蔬菜分类 |
|classification/resnet50.py | 图像分类ResNet50 | 蔬菜分类 |
|detection/faster_rcnn_r50_fpn.py | 目标检测FasterRCNN | 昆虫检测 |
|detection/mask_rcnn_f50_fpn.py | 实例分割MaskRCNN | 垃圾分拣 |
|segmentation/deeplabv3p.py | 语义分割DeepLabV3| 视盘分割 |
|segmentation/unet.py | 语义分割UNet | 视盘分割 |

## 开始训练
在安装PaddleX后，使用如下命令开始训练
```
python classification/mobilenetv2.py
```
