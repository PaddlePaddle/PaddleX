# 使用教程——训练模型

本目录下整理了使用PaddleX训练模型的示例代码，代码中均提供了示例数据的自动下载，并均使用单张GPU卡进行训练。

|代码 | 模型任务 | 数据 |
|------|--------|---------|
|image_classification/alexnet.py | 图像分类AlexyNet | 蔬菜分类 |
|image_classification/shufflenetv2.py | 图像分类ShuffleNetV2 | 蔬菜分类 |
|image_classification/mobilenetv3_small.py | 图像分类MobileNetV3_small | 蔬菜分类 |
|image_classification/mobilenetv3_large_w_custom_optimizer.py | 图像分类MobileNetV3_large | 蔬菜分类 |
|image_classification/resnet50_vd_ssld.py | 图像分类ResNet50_vd_ssld | 蔬菜分类 |
|image_classification/darknet53.py | 图像分类DarkNet53 | 蔬菜分类 |
|image_classification/xception41.py | 图像分类Xception41 | 蔬菜分类 |
|image_classification/densenet121.py | 图像分类DenseNet121 | 蔬菜分类 |
|object_detection/faster_rcnn_r50_fpn.py | 目标检测FasterRCNN | 昆虫检测 |
|object_detection/ppyolo.py | 目标检测PPYOLO | 昆虫检测 |
|object_detection/ppyolotiny.py | 目标检测PPYOLOTiny | 昆虫检测 |
|object_detection/ppyolov2.py | 目标检测PPYOLOv2 | 昆虫检测 |
|object_detection/yolov3_darknet53.py | 目标检测YOLOv3 | 昆虫检测 |
|semantic_segmentation/deeplabv3p_resnet50_vd.py | 语义分割DeepLabV3 | 视盘分割 |
|semantic_segmentation/bisenetv2.py | 语义分割BiSeNetV2 | 视盘分割 |
|semantic_segmentation/fast_scnn.py | 语义分割FastSCNN | 视盘分割 |
|semantic_segmentation/hrnet.py | 语义分割HRNet | 视盘分割 |
|semantic_segmentation/unet.py | 语义分割UNet | 视盘分割 |

可参考API接口说明了解示例代码中的API：
* [数据集读取API](../../docs/apis/datasets.md)
* [数据预处理和数据增强API](../../docs/apis/transforms/transforms.md)
* [模型API/模型加载API](../../docs/apis/models/README.md)
* [预测结果可视化API](../../docs/apis/visualize.md)


# 环境准备

- [PaddlePaddle安装](https://www.paddlepaddle.org.cn/install/quick)
* 版本要求：PaddlePaddle>=2.1.0
* **运行MaskRCNN要求PaddlePaddle>=2.1.1**

- [PaddleX安装](../../docs/install.md)

## 开始训练
* 在安装PaddleX后，使用如下命令开始训练，代码会自动下载训练数据, 并均使用单张GPU卡进行训练。

```commandline
export CUDA_VISIBLE_DEVICES=0
python image_classification/mobilenetv3_small.py
```

* 若需使用多张GPU卡进行训练，例如使用2张卡时执行：

```commandline
python -m paddle.distributed.launch --gpus 0,1 image_classification/mobilenetv3_small.py
```
使用多卡时，参考[训练参数调整](../../docs/parameters.md)调整学习率和批量大小。


## VisualDL可视化训练指标
在模型训练过程，在`train`函数中，将`use_vdl`设为True，则训练过程会自动将训练日志以VisualDL的格式打点在`save_dir`（用户自己指定的路径）下的`vdl_log`目录，用户可以使用如下命令启动VisualDL服务，查看可视化指标
```commandline
visualdl --logdir output/mobilenetv3_small/vdl_log --port 8001
```

服务启动后，使用浏览器打开 https://0.0.0.0:8001 或 https://localhost:8001


## 版本升级


如果训练脚本、API和模型是之前使用PaddleX==1.x得到的，然后对PaddleX进行升级到2.0，此时再次运行之前的脚本会报错并提示`Your running script needs PaddleX<2.0.0, please refer to https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train#%E7%89%88%E6%9C%AC%E5%8D%87%E7%BA%A7 to solve this issue`，**这是因为PaddleX升级到2.0后，对部分训练API的设定做了修改，无法对1.x版本的PaddleX训练脚本做兼容**。

如果需要使用PaddleX版本低于2.0.0的训练脚本、API和模型，请将PaddleX回退到1.3.11版本。如果想继续使用PaddleX 2.0，那么直接运行本目录下的训练脚本开始体验吧。
