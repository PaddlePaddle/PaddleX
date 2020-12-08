# Tutorial - Training model

The example codes of the PaddleX training model are in this directory. The codes provide the automatic download of the example data. A single GPU card is used for training.

| Codes | Model task | Data |
|------|--------|---------|
| image_classification/alexnet.py | Image Classification AlexyNet | Vegetable Classification |
| image_classification/mobilenetv2.py | Image Classification MobileNetV2 | Vegetable Classification |
| image_classification/mobilenetv3_small_ssld.py | Image Classification MobileNetV3_small_ssld | Vegetable Classification |
| image_classification/resnet50_vd_ssld.py | Image Classification ResNet50_vd_ssld | Vegetable Classification |
| image_classification/shufflenetv2.py | Image Classification ShuffleNetV2 | Vegetable Classification |
| object_detection/faster_rcnn_hrnet_fpn.py | Object Detection FasterRCNN | Insect Detection |
| object_detection/faster_rcnn_r18_fpn.py | Object Detection FasterRCNN | Insect Detection |
| object_detection/faster_rcnn_r50_fpn.py | Object Detection FasterRCNN | Insect Detection |
| object_detection/ppyolo.py | Object Detection PPYOLO | Insect Detection |
| object_detection/yolov3_darknet53.py | Object Detection YOLOv3 | Insect Detection |
| object_detection/yolov3_mobilenetv1.py | Object Detection YOLOv3 | Insect Detection |
| object_detection/yolov3_mobilenetv3.py | Object Detection YOLOv3 | Insect Detection |
| instance_segmentation/mask_rcnn_hrnet_fpn.py | Instance Segmentation MaskRCNN | Dudu Sorting |
| instance_segmentation/mask_rcnn_r18_fpn.py | Instance Segmentation MaskRCNN | Dudu Sorting |
| instance_segmentation/mask_rcnn_f50_fpn.py | Instance Segmentation MaskRCNN | Dudu Sorting |
| semantic_segmentation/deeplabv3p_mobilenetv2.py | Semantic Segmentation DeepLabV3 | Video Segmentation |
| semantic_segmentation/deeplabv3p_mobilenetv2.py | Semantic Segmentation DeepLabV3 | Video Segmentation |
| semantic_segmentation/deeplabv3p_mobilenetv2_x0.25.py | Semantic Segmentation DeepLabV3 | Video Segmentation |
| semantic_segmentation/deeplabv3p_xception65.py | Semantic Segmentation DeepLabV3 | Video Segmentation |
| semantic_segmentation/fast_scnn.py | Semantic Segmentation FastSCNN | Video Segmentation |
| semantic_segmentation/hrnet.py | Semantic Segmentation HRNet | Video Segmentation |
| semantic_segmentation/unet.py | Semantic Segmentation UNet | Video Segmentation |

## Start training
After installing PaddleX, start the training by running the following command:
```
python image_classification/mobilenetv2.py
```
