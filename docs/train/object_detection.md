# Object detection

## Introduction

Currently, PaddleX provides FasterRCNN and YOLOv3 detection structures and various backbone models to meet the requirements of developers for different scenarios and performances.

- **Box MMAP**: Model test precision on the COCO dataset
- **Inference speed**: Inference time for a single image (preprocessing and postprocessing excluded)
- "-" indicates that the indexes are not updated temporarily

| Model (Click to obtain codes) | Box MMAP | Model Size | GPU Inference Speed | Arm Inference Speed | Note |
| :----------------  | :------- | :------- | :---------  | :---------  | :-----    |
| [YOLOv3-MobileNetV1](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_mobilenetv1.py) | 29.3% | 99.2 MB | 15.442 ms | - | The model is small, has a fast inference speed and applies to low-performance or mobile devices |
|
| [YOLOv3-MobileNetV3](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_mobilenetv3.py) | 31.6% | 100.7 MB | 143.322 ms | - | The model is small and has an advantageous inference speed on the mobile terminal |
| [YOLOv3-DarkNet53](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/yolov3_darknet53.py) | 38.9% | 249.2 MB | 42.672 ms | - | The model is large, has a fast inference speed and applies to the server |
| [PPYOLO](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/ppyolo.py) | 45.9% | 329.1 MB | - | - | The model is large, has the faster inference speed than YOLOv3-DarkNet53 and applies to the server |
| [FasterRCNN-ResNet50-FPN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/faster_rcnn_r50_fpn.py) | 37.2% | 167.7 MB | 197.715 ms | - | The model has a high precision and applies to server deployment |
| [FasterRCNN-ResNet18-FPN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/faster_rcnn_r18_fpn.py) | 32.6% | 173.2 MB | - | - | The model has a high precision and applies to server deployment |
| [FasterRCNN-HRNet-FPN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/object_detection/faster_rcnn_hrnet_fpn.py) | 36.0% | 115. MB | 81.592 ms | - | The model has a high precision and a fast inference speed and applies to server deployment |


## Start training

Save and run codes locally (The code downloading links are located in the table above) and **codes automatically download training data and start training**.  If codes are saved as `yolov3_mobilenetv1.py`, execute the following command to start training:

```
python yolov3_mobilenetv1.py
```


## Related document

- [**Important**] Adjust training parameters according to your machine environment and data, adjust training parameters? Understand the role of training parameters in PaddleX first. [——>>Portal] (../appendix/parameters.md)
- [**Useful**] There are no machine resources? Use a free AIStudio GPU resource: online training model. [——>>Portal] (https://aistudio.baidu.com/aistudio/projectdetail/450925)
- [**Extension**] For more object detection models, refer to the [PaddleX model library](../appendix/model_zoo.md) and the [API operation document](../apis/models/detection.md). 