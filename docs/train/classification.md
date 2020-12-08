# Image classification

## Introduction

PaddleX provides a total of more than 20 image classification models to meet the requirements of developers for different scenarios.

- **Top1 precision**: Model test precision on the ImageNet dataset
- **Inference speed**: Inference time for a single image (preprocessing and postprocessing excluded)
- "-" indicates that the indexes are not updated temporarily

| Model (Click to obtain codes) | Top1 precision | Model size | GPU Inference Speed | Arm Inference Speed | Note |
| :----------------  | :------- | :------- | :---------  | :---------  | :-----    |
| [MobileNetV3_small_ssld](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv3_small_ssld.py) | 71.3% | 21.0 MB | 6.809 ms | - | The model is small, has a fast inference speed and applies to low-performance or mobile devices |
| [MobileNetV2](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py) | 72.2% | 14.0 MB | 4.546 ms | - | The model is small, has a fast inference speed and applies to low-performance or mobile devices |
| [ShuffleNetV2](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/shufflenetv2.py) | 68.8% | 9.0 MB | 6.101 ms | - | The model has a small volume and a fast inference speed and applies to low-performance or mobile terminal devices |
| [ResNet50_vd_ssld](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/resnet50_vd_ssld.py) | 82.4% | 102.8 MB | 9.058 ms | - | The model has a high precision and applies to server deployment |


## Start training

Save and run codes locally (The code downloading links are located in the table above) and **codes automatically download training data and start training*. If codes are saved as `mobilenetv3_small_ssld.py`, execute the following command to start training:

```
python mobilenetv3_small_ssld.py
```


## Related document

- [**Important**] Adjust training parameters according to your machine environment and data, adjust training parameters? Understand the role of training parameters in PaddleX first. [——>>Portal] (../appendix/parameters.md)
- [**Useful**] There are no machine resources? Use a free AIStudio GPU resource: online training model. [——>>Portal] (https://aistudio.baidu.com/aistudio/projectdetail/450925)
- [**Extension**] For more image classification models, refer to the [PaddleX model library](../appendix/model_zoo.md) and the [API operation document](../apis/models/classification.md).
