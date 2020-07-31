# 使用教程——训练模型

本目录下整理了使用PaddleX训练模型的示例代码，代码中均提供了示例数据的自动下载，并均使用单张GPU卡进行训练。

|代码 | 模型任务 | 数据 |
|------|--------|---------|
|image_classification/alexnet.py | 图像分类AlexyNet | 蔬菜分类 |
|image_classification/mobilenetv2.py | 图像分类MobileNetV2 | 蔬菜分类 |
|image_classification/mobilenetv3_small_ssld.py | 图像分类MobileNetV3_small_ssld | 蔬菜分类 |
|image_classification/resnet50_vd_ssld.py | 图像分类ResNet50_vd_ssld | 蔬菜分类 |
|image_classification/shufflenetv2.py | 图像分类ShuffleNetV2 | 蔬菜分类 |
|object_detection/faster_rcnn_hrnet_fpn.py | 目标检测FasterRCNN | 昆虫检测 |
|object_detection/faster_rcnn_r18_fpn.py | 目标检测FasterRCNN | 昆虫检测 |
|object_detection/faster_rcnn_r50_fpn.py | 目标检测FasterRCNN | 昆虫检测 |
|object_detection/yolov3_darknet53.py | 目标检测YOLOv3 | 昆虫检测 |
|object_detection/yolov3_mobilenetv1.py | 目标检测YOLOv3 | 昆虫检测 |
|object_detection/yolov3_mobilenetv3.py | 目标检测YOLOv3 | 昆虫检测 |
|instance_segmentation/mask_rcnn_hrnet_fpn.py | 实例分割MaskRCNN | 小度熊分拣 |
|instance_segmentation/mask_rcnn_r18_fpn.py | 实例分割MaskRCNN | 小度熊分拣 |
|instance_segmentation/mask_rcnn_f50_fpn.py | 实例分割MaskRCNN | 小度熊分拣 |
|semantic_segmentation/deeplabv3p_mobilenetv2.py | 语义分割DeepLabV3 | 视盘分割 |
|semantic_segmentation/deeplabv3p_mobilenetv2_x0.25.py | 语义分割DeepLabV3 | 视盘分割 |
|semantic_segmentation/deeplabv3p_xception65.py | 语义分割DeepLabV3 | 视盘分割 |
|semantic_segmentation/fast_scnn.py | 语义分割FastSCNN | 视盘分割 |
|semantic_segmentation/hrnet.py | 语义分割HRNet | 视盘分割 |
|semantic_segmentation/unet.py | 语义分割UNet | 视盘分割 |

## 开始训练
在安装PaddleX后，使用如下命令开始训练
```
python image_classification/mobilenetv2.py
```
