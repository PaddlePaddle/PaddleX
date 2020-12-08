# Semantic segmentation

## Introduction

Currently, PaddleX provides DeepLabv3p, UNet, HRNet and FastSCNN with semantic segmentation structures and various backbone models to meet the requirements of developers for different scenarios and performance.

- **mIoU**: Model test precision on the CityScape dataset
- **Inference speed**: Inference time for a single image (preprocessing and postprocessing excluded)
- "-" indicates that the indexes are not updated temporarily

| Model (Click to obtain codes) | mIOU | Model Size | GPU Inference Speed | Arm Inference Speed | Note |
| :----------------  | :------- | :------- | :---------  | :---------  | :-----    |
| [DeepLabv3p-MobileNetV2-x0.25](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/deeplabv3p_mobilenetv2_x0.25.py) | -- | 2.9 MB | - | - | The model is small, has a fast inference speed and applies to low-performance or mobile devices |
| [DeepLabv3p-MobileNetV2-x1.0](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/deeplabv3p_mobilenetv2.py) | 69.8% | 11 MB | - | - | The model is small, has a fast inference speed and applies to low-performance or mobile devices |
| [DeepLabv3_MobileNetV3_large_x1_0_ssld](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/deeplabv3p_mobilenetv3_large_ssld.py) | 73.28% | 9.3 MB | - | - | The model is small, has a fast inference speed and a high precision and applies to low-performance or mobile devices |
| [DeepLabv3p-Xception65](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/deeplabv3p_xception65.py) | 79.3% | 158 MB | - | - | The model is large, has a high precision and applies to the server |
| [UNet](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/unet.py) | - | 52 MB | - | - | The model is large, has a high precision and applies to the server |
| [HRNet](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/hrnet.py) | 79.4% | 37 MB | - | - | The model is large, has a high model precision and applies to server deployment |
| [FastSCNN](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/semantic_segmentation/fast_scnn.py) | - | 4.5 MB | - | - | The model is small, has a fast inference speed and applies to low-performance or mobile devices |


## Start training

Save and run codes locally (The code downloading links are located in the table above) and **codes automatically download training data**. Start training. If codes are saved as `deeplabv3p_mobilenetv2_x0.25.py`, execute the following command to start training:
```
python deeplabv3p_mobilenetv2_x0.25.py
```


## Related document

- [**Important**] Adjust training parameters according to your machine environment and data, adjust training parameters? Understand the role of training parameters in PaddleX first. ——>>[Portal] (../appendix/parameters.md)
- [**Useful**] There are no machine resources? Use a free AIStudio GPU resource: online training model. ——>>[Portal] (https://aistudio.baidu.com/aistudio/projectdetail/450925)
- [**Extension**] For more semantic segmentation models, refer to the [PaddleX model library](../appendix/model_zoo.md) and the [API operation document](../apis/models/semantic_segmentation.md).
