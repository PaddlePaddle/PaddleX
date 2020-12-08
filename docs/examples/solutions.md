# Introduction to the PaddleX model

PaddleX provides rich model algorithms for four vision tasks: image classification, object detection, instance segmentation and semantic segmentation, allowing users to choose the appropriate model as required in real scenarios.

## Image classification
The image classification task refers to the input of an image for the model to predict the category of the image, for example, identifying it as a landscape, animal, car, and so on.

![](./images/image_classification.png)

For the image classification task, PaddleX provides Baidu's improved model for different application scenarios. See the following table:

> * GPU prediction speed is obtained based on the evaluation environment of the T4 machine by measuring 500 times under FP32+TensorRT configuration (removing the warmup time of the first 10 times).
> * CPU prediction speed is based on the evaluation environment of Snapdragon 855 (SD855).
> * Top 1 accuracy is evaluated on the ImageNet-1000 dataset.

| Model | Model Feature | Storage Size | GPU time(ms) bs=1 | SD855 time(ms) bs=1 | Top1 Accuracy |
| :--------- | :------  | :---------- | :-----------| :-------------  | :--- |
| MobileNetV3_small_ssld | Lightweight and high speed, suitable for real-time mobile scenes in high speed. | 12.5MB | - | 6.5463 | 71.3.0% |
| ShuffleNetV2 | Lightweight model with relatively low precision, suitable for real-time mobile scenarios requiring smaller storage size | 10.2MB | - | 10.941 | 68.8% |
| MobileNetV3_large_ssld | Lightweight model with little storage advantage, moderate performance in terms of speed and precision, suitable for mobile scenarios | 22.8MB | - | 19.30835 | 79.0% |
| MobileNetV2 | Lightweight model, suitable for mobile scenarios by using GPU predictions | 15.0MB | - | 23.317699 | 72.2% |
| ResNet50_vd_ssld | High precision models with short prediction time, suitable for most server scenarios | 103.5MB | 3.47712 | - | 82.4% |
| ResNet101_vd_ssld | Ultra-high precision model with relatively long prediction time, suitable for server scenarios with large data volumes. | 180.5MB | 6.11704 | - | 83.7% |
| Xception65 | Ultra-high precision model with longer prediction time and higher precision when processing large data volumes, suitable for server scenarios | 161.6MB | 7.26158 | - | 80.30% |

In addition to the above models, PaddleX supports nearly 20 image classification models. For the rest of the models, refer to the [PaddleX model library]. (../appendix/model_zoo.md)


## Object detection
Object detection task means that the model identifies the position of the object in the input image (framed by a rectangle and marked the position of the frame) and the object class. For example, detect cosmetic defects in the quality control of parts of mobile phones.

![](./images/object_detection.png)

For object detection, PaddleX provides the mainstream YOLOv3 model and the Faster-RCNN model in different application scenarios. See the following table:

> * GPU prediction speed is obtained based on the evaluation environment of Tesla V100 by measuring 100 times under FP32+TensorRT configuration (removing the warmup time of the first 10 times).
> * Box mmAP is the evaluation result on the MSCOCO dataset.

| Model | Model Feature | Storage Size | GPU time(ms) bs=1 | Box mmAP |
| :------- | :-------  | :---------  | :---------- | :--- |
| YOLOv3-MobileNetV3_large | Suitable for mobile scenarios where high-speed prediction is required | 100.7 MB | - | 31.6 |
| YOLOv3-MobileNetV1 | Relatively low precision, suitable for server-side scenarios that require high-speed prediction. | 99.2 MB | 11.834 | 29.3 |
| YOLOv3-DarkNet53 | Good performance in terms of prediction speed and model precision for most server-side scenarios | 249.2 MB | 20.252 | 38.9 |
| PPYOLO | Better prediction speed and model precision than YOLOv3-DarkNet53, suitable for most server-side scenarios | 329.1 MB | - | 45.9 |
| FasterRCNN-ResNet50-FPN | Classic two-stage detector with relatively slow prediction speed for server-side scenarios that attach importance to the model precision | 167 MB | 24.758 | 37.2 |
| FasterRCNN - HRNet_W18-FPN | Suitable for server-side scenes that are sensitive to image resolution and require more details of object prediction | 115.5MB | - | 36 |
| FasterRCNN-ResNet101_vd-FPN | Ultra-high precision model with longer prediction time and higher precision when processing large data volumes, suitable for server scenarios | 244.3MB | 29.969 | 40.5 |

In addition to the above models, YOLOv3 and Faster RCNN also support other backbones. For details, refer to the [PaddleX model library] (../appendix/model_zoo.md). 

### Instance segmentation
In the object detection, the model identifies the location of the object in the image and the category of the object. Instance segmentation, with pixel-level classification on the basis of object detection, identifies the pixels within the frame that belong to the target object.

![](./images/instance_segmentation.png)

PaddleX currently provides the instance segmentation MaskRCNN model, which supports 5 different backbone networks. Refer to the [PaddleX model library] for more details (../appendix/model_zoo.md).

> * GPU prediction speed is obtained based on the evaluation environment of Tesla V100 by measuring 100 times under FP32+TensorRT configuration (removing the warmup time of the first 10 times).
> * Box mmAP is the evaluation result on the MSCOCO dataset.


| Model | Model Feature | Storage Size | GPU time(ms) bs=1 | Box mmAP | Seg mmAP |
| :---- | :------- | :---------- | :---------- | :--- |:--- |
| MaskRCNN - HRNet_W18-FPN | Suitable for server-side scenes that are sensitive to image resolution and require more details of object prediction | 143.9MB | -- | 38.2 | 33.4 |
| MaskRCNN-ResNet50-FPN | High precision, suitable for most server-side scenarios | 177.7M | 83.567 | 38.7 | 34.7 |
| MaskRCNN-ResNet101_vd-FPN | High precision with longer prediction time, suitable for server-side scenarios in the processing of larger data volumes with higher precision | 253.7M | 97.647 | 41.4 | 36.8 |

## Semantic segmentation
Semantic segmentation is used to perform pixel-level classification of images, and is applied in scenarios such as portrait classification and remote sensing image recognition.

![](./images/semantic_segmentation.png)

For semantic segmentation, PaddleX also provides different model options for different scenarios. See the following table:

> * mIoU is evaluated on the Cityscapes dataset.


| Model | Model Feature | Storage Size | Prediction Speed (milliseconds) | mIOU |
| :---- | :------- | :---------- | :--- | :--- |
| DeepLabv3p-MobileNetV2_x1.0 | Lightweight model, suitable for mobile scenarios | 14.7MB | - | 69.8% |
| DeepLabv3-MobileNetV3_large_x1_0_ssld | Lightweight model, suitable for mobile scenarios | 9.3 MB | - | 73.28% |
| HRNet_W18_Small_v1 | Lightweight and high speed, suitable for mobile scenes | - | - | - |
| FastSCNN | Lightweight and high speed, suitable for mobile or server-side scenarios that require high speed prediction. | 9.8MB | - | 69.64 |
| HRNet_W18 | High-precision model, suitable for server-side scenarios that are sensitive to image resolution and require higher prediction of target details. | 77.3MB | - | 79.36 |
| DeepLabv3p-Xception65 | High precision with longer prediction time, suitable for larger data volume with high precision in server scenarios with complex background | 329.3MB | - | 79.3% |
