# 加载模型预测

PaddleX可以使用`load_model`接口加载模型（包括训练过程中保存的模型，导出的部署模型，量化模型以及裁剪的模型）进行预测，同时PaddleX中也内置了一系列的可视化工具函数，帮助用户方便地检查模型的效果。

## 图像分类
```
import paddlex as pdx
model = pdx.load_model('./mobilenetv2')
result = model.predict('./mobilenetv2/test.jpg')
print("Predict Result: ", result)
```


## 目标检测和实例分割
```
import paddlex as pdx
test_jpg = './xiaoduxiong_epoch_12/test.jpg'
model = pdx.load_model('./xiaoduxiong_epoch_12')
result = model.predict(test_jpg)
pdx.det.visualize(test_jpg, result, thresh=0.5, save_dir='./')
```
在上述示例代码中，通过调用`paddlex.det.visualize`可以对目标检测/实例分割的预测结果进行可视化，可视化的结果保存在`save_dir`下。
> 注意：目标检测和实例分割模型在调用`predict`接口得到的结果需用户自行过滤低置信度结果，在`paddlex.det.visualize`接口中，我们提供了`thresh`用于过滤，置信度低于此值的结果将被过滤，不会可视化。


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
| 图像分类 | [MobileNetV3_small_ssld]() | ImageNet | xxMB | Accuracy  |             |
| 图像分类 | [ResNet50_vd_ssld]()  | ImageNet  | xxMB  | Accuracy  |              |
| 目标检测 | [FasterRCNN-ResNet50-FPN]() | MSCOCO | xxMB     |    Box MAP  |                |
| 目标检测 | [YOLOv3-MobileNetV1]()    | MSCOCO | xxMB      | Box MAP    |                 |
| 目标检测 | [YOLOv3-DarkNet53]()      | MSCOCO | xxMB      | Box MAP    |                 |
| 实例分割 | [MaskRCNN-ResNet50-FPN]()  | MSCOCO | xxMB     | Box MAP/Seg MAP |            |
| 语义分割 | [DeepLabv3p-Xception65]()  | 人像分割 | xxMB     | mIoU        |      -          |
| 语义分割 | [HRNet_w18_small]()           | 人像分割   | xxMB   | mIou       |        -           |

PaddleX的`load_model`接口可以满足用户一般的模型调研需求，如若为更高性能的预测部署，可以参考如下文档

- [服务端Python部署]()  
- [服务端C++部署]()


