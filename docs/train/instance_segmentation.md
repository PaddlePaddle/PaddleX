# Instance segmentation

## Introduction

Currently, PaddleX provides a MaskRCNN with instance segmentation model structure and various backbone models to meet the requirements of developers for different scenarios and performance.

- **Box MMAP/Seg MMAP**: Model test precision on the COCO dataset
- **Inference speed**: Inference time for a single image (preprocessing and postprocessing excluded)
- "-" indicates that the indexes are not updated temporarily

| Model (Click to obtain codes) | Box MMAP/Seg MMAP | Model Size | GPU Inference Speed | Arm Inference Speed | Note |
| :----------------  | :------- | :------- | :---------  | :---------  | :-----    |
| [MaskRCNN-ResNet50-FPN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py) | 38.7% / 34.7% | 177.7 MB | 160.185 ms | - | The model has a high precision and applies to server deployment |
| [MaskRCNN-ResNet18-FPN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_r18_fpn.py) | 33.6%/30.5% | 189.1 MB | - | - | The model has a high precision and applies to server deployment |
| [MaskRCNN-HRNet-FPN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/instance_segmentation/mask_rcnn_hrnet_fpn.py) | 38.7% / 34.7% | 120.7 MB | - | - | The model has a high precision and a fast inference speed and applies to server deployment |


## Start training

Save and run codes locally (The code downloading links are located in the table above) and **codes automatically download training data and start training**. If codes are saved as `mask_rcnn_r50_fpn.py`, execute the following command to start training:

```
python mask_rcnn_r50_fpn.py
```

## Related document

- [**Important**] Adjust training parameters according to your machine environment and data, adjust training parameters? Understand the role of training parameters in PaddleX first. [——>>Portal](../appendix/parameters.md)
- [**Useful**] There are no machine resources? Use a free AIStudio GPU resource: online training model. [——>>Portal](https://aistudio.baidu.com/aistudio/projectdetail/450925)
- [**Extension**] For more instance segmentation models, refer to the [PaddleX model library] (../appendix/model_zoo.md) and the [API operation document](../apis/models/instance_segmentation.md).
