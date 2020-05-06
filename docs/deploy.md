# 模型预测部署

本文档指引用户如何采用更高性能地方式来部署使用PaddleX训练的模型。使用本文档模型部署方式，会在模型运算过程中，对模型计算图进行优化，同时减少内存操作，相对比普通的paddlepaddle模型加载和预测方式，预测速度平均可提升1倍，具体各模型性能对比见[预测性能对比](#预测性能对比)

## 服务端部署

### 导出inference模型

在服务端部署的模型需要首先将模型导出为inference格式模型，导出的模型将包括`__model__`、`__params__`和`model.yml`三个文名，分别为模型的网络结构，模型权重和模型的配置文件（包括数据预处理参数等等）。在安装完PaddleX后，在命令行终端使用如下命令导出模型到当前目录`inferece_model`下。

> 可直接下载垃圾检测模型测试本文档的流程[garbage_epoch_12.tar.gz](https://bj.bcebos.com/paddlex/models/garbage_epoch_12.tar.gz)

```
paddlex --export_inference --model_dir=./garbage_epoch_12 --save_dir=./inference_model
```

### Python部署
PaddleX已经集成了基于Python的高性能预测接口，在安装PaddleX后，可参照如下代码示例，进行预测。相关的接口文档可参考[paddlex.deploy](apis/deploy.md)
> 点击下载测试图片 [garbage.bmp](https://bj.bcebos.com/paddlex/datasets/garbage.bmp)
```
import paddlex as pdx
predictorpdx.deploy.create_predictor('./inference_model')
result = predictor.predict(image='garbage.bmp')
```

### C++部署

C++部署方案位于目录`deploy/cpp/`下，且独立于PaddleX其他模块。该方案支持在 Windows 和 Linux 完成编译、二次开发集成和部署运行。具体使用方法和编译：

- Linux平台：[linux](deploy_cpp_linux.md)
- window平台：[windows](deploy_cpp_win_vs2019.md)

### 预测性能对比

#### 测试环境

- CUDA 9.0
- CUDNN 7.5
- PaddlePaddle 1.71
- GPU: Tesla P40
- AnalysisPredictor 指采用Python的高性能预测方式
- Executor 指采用paddlepaddle普通的python预测方式
- Batch Size均为1，耗时单位为ms/image，只计算模型运行时间，不包括数据的预处理和后处理

| 模型 | AnalysisPredictor耗时 | Executor耗时 | 输入图像大小 |
| :---- | :--------------------- | :------------ | :------------ |
| resnet50 | 4.84 | 7.57 | 224*224 |
| mobilenet_v2 | 3.27 | 5.76 | 224*224 |
| unet | 22.51 | 34.60 |513*513 |
| deeplab_mobile | 63.44 | 358.31 |1025*2049 |
| yolo_mobilenetv2 | 15.20 | 19.54 |  608*608 |
| faster_rcnn_r50_fpn_1x | 50.05 | 69.58 |800*1088 |
| faster_rcnn_r50_1x | 326.11 | 347.22 | 800*1067 |
| mask_rcnn_r50_fpn_1x | 67.49 | 91.02 | 800*1088 |
| mask_rcnn_r50_1x | 326.11 | 350.94 | 800*1067 |

## 移动端部署

> Lite模型导出正在集成中，即将开源...
