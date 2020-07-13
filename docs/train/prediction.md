# 加载模型预测

PaddleX可以使用`paddlex.load_model`接口加载模型（包括训练过程中保存的模型，导出的部署模型，量化模型以及裁剪的模型）进行预测，同时PaddleX中也内置了一系列的可视化工具函数，帮助用户方便地检查模型的效果。

> 注意：使用`paddlex.load_model`接口加载仅用于模型预测，如需要在此模型基础上继续训练，可以将该模型作为预训练模型进行训练，具体做法是在训练代码中，将train函数中的`pretrain_weights`参数指定为预训练模型路径。

## 图像分类

> [点击下载](https://bj.bcebos.com/paddlex/models/mobilenetv3_small_ssld_imagenet.tar.gz)如下示例代码中模型  
```
import paddlex as pdx
test_jpg = 'mobilenetv3_small_ssld_imagenet/test.jpg'
model = pdx.load_model('mobilenetv3_small_ssld_imagenet')
result = model.predict(test_jpg)
print("Predict Result: ", result)
```
结果输入如下
```
Predict Result: [{'category_id': 21, 'category': 'killer_whale', 'score': 0.8262267}]
```
测试图片如下


- 分类模型predict接口[说明文档](../apis/models/classification.html#predict)


## 目标检测

> [点击下载](https://bj.bcebos.com/paddlex/models/yolov3_mobilenetv1_coco.tar.gz)如下示例代码中模型  
```
import paddlex as pdx
test_jpg = 'yolov3_mobilenetv1_coco/test.jpg'
model = pdx.load_model('yolov3_mobilenetv1_coco')

# predict接口并未过滤低置信度识别结果，用户根据需求按score值进行过滤
result = model.predict(test_jpg)

# 可视化结果存储在./visualized_test.jpg, 见下图
pdx.det.visualize(test_jpg, result, threshold=0.3, save_dir='./')
```
- YOLOv3模型predict接口[说明文档](../apis/models/detection.html#predict)
- 可视化pdx.det.visualize接口[说明文档](../apis/visualize.html#paddlex-det-visualize)
> 注意：目标检测和实例分割模型在调用`predict`接口得到的结果需用户自行过滤低置信度结果，在`paddlex.det.visualize`接口中，我们提供了`threshold`用于过滤，置信度低于此值的结果将被过滤，不会可视化。
![](./images/yolo_predict.jpg)

## 实例分割

> [点击下载](https://bj.bcebos.com/paddlex/models/mask_r50_fpn_coco.tar.gz)如下示例代码中模型  
```
import paddlex as pdx
test_jpg = 'mask_r50_fpn_coco/test.jpg'
model = pdx.load_model('mask_r50_fpn_coco')

# predict接口并未过滤低置信度识别结果，用户根据需求按score值进行过滤
result = model.predict(test_jpg)

# 可视化结果存储在./visualized_test.jpg, 见下图
pdx.det.visualize(test_jpg, result, threshold=0.5, save_dir='./')
```
- MaskRCNN模型predict接口[说明文档](../apis/models/instance_segmentation.html#predict)
- 可视化pdx.det.visualize接口[说明文档](../apis/visualize.html#paddlex-det-visualize)

> 注意：目标检测和实例分割模型在调用`predict`接口得到的结果需用户自行过滤低置信度结果，在`paddlex.det.visualize`接口中，我们提供了`threshold`用于过滤，置信度低于此值的结果将被过滤，不会可视化。
![](./images/mask_predict.jpg)

## 语义分割
```
import paddlex as pdx
test_jpg = './deeplabv3p_mobilenetv2_coco/test.jpg'
model = pdx.load_model('./deeplabv3p_mobilenetv2_coco')
result = model.predict(test_jpg)
pdx.seg.visualize(test_jpg, result, weight=0.0, save_dir='./')
```

在上述示例代码中，通过调用`paddlex.seg.visualize`可以对语义分割的预测结果进行可视化，可视化的结果保存在`save_dir`下。其中`weight`参数用于调整预测结果和原图结果融合展现时的权重，0.0时只展示预测结果mask的可视化，1.0时只展示原图可视化。


## 公开数据集训练模型下载

PaddleX提供了部分公开数据集上训练好的模型，用户可以直接下载后参照本文档加载使用。

| 类型 |     模型(点击下载)     |     数据集    |     大小     |     指标    |    指标数值    |
|:--- | :----------  | :-----------  | :----------  | :---------- | :------------- |
| 图像分类 | [MobileNetV3_small_ssld](https://bj.bcebos.com/paddlex/models/mobilenetv3_small_ssld_imagenet.tar.gz) | ImageNet | 13MB | Accuracy  |     71.3%        |
| 图像分类 | [ResNet50_vd_ssld](https://bj.bcebos.com/paddlex/models/resnet50_vd_ssld_imagenet.tar.gz)  | ImageNet  | 110MB  | Accuracy  |   82.4%       |
| 目标检测 | [FasterRCNN-ResNet50-FPN](https://bj.bcebos.com/paddlex/models/faster_r50_fpn_coco.tar.gz) | MSCOCO | 179MB     |    Box MAP  |       37.7%     |
| 目标检测 | [YOLOv3-MobileNetV1](https://bj.bcebos.com/paddlex/models/yolov3_mobilenetv1_coco.tar.gz)    | MSCOCO | 106MB      | Box MAP    |      29.3%      |
| 目标检测 | [YOLOv3-DarkNet53](https://bj.bcebos.com/paddlex/models/yolov3_darknet53_coco.tar.gz)      | MSCOCO | 266MMB      | Box MAP    |      34.8%      |
| 目标检测 | [YOLOv3-MobileNetV3](https://bj.bcebos.com/paddlex/models/yolov3_mobilenetv3_coco.tar.gz)      | MSCOCO | 101MB      | Box MAP    |      31.6%      |
| 实例分割 | [MaskRCNN-ResNet50-FPN](https://bj.bcebos.com/paddlex/models/mask_r50_fpn_coco.tar.gz)  | MSCOCO | 193MB     | Box MAP/Seg MAP |   38.7% / 34.7%     |
| 语义分割 | DeepLabv3p-Xception65  | 人像分割 | -     | mIoU        |      -          |
| 语义分割 | HRNet_w18_small           | 人像分割   | -   | mIou       |        -           |

PaddleX的`load_model`接口可以满足用户一般的模型调研需求，如若为更高性能的预测部署，可以参考如下文档

- [服务端Python部署](../deploy/server/python.md)  
- [服务端C++部署](../deploy/server/cpp/index.html)
