# Python部署
PaddleX已经集成了基于Python的高性能预测接口，在安装PaddleX后，可参照如下代码示例，进行预测。相关的接口文档可参考[paddlex.deploy](../../../apis/deploy.md)

## 导出inference模型

在服务端部署的模型需要首先将模型导出为inference格式模型，导出的模型将包括`__model__`、`__params__`和`model.yml`三个文名，分别为模型的网络结构，模型权重和模型的配置文件（包括数据预处理参数等等）。在安装完PaddleX后，在命令行终端使用如下命令导出模型到当前目录`inferece_model`下。
> 可直接下载小度熊分拣模型测试本文档的流程[xiaoduxiong_epoch_12.tar.gz](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz)

```
paddlex --export_inference --model_dir=./xiaoduxiong_epoch_12 --save_dir=./inference_model
```

使用TensorRT预测时，需指定模型的图像输入shape:[w,h]。
**注**：
- 分类模型请保持于训练时输入的shape一致。
- 指定[w,h]时，w和h中间逗号隔开，不允许存在空格等其他字符

```
paddlex --export_inference --model_dir=./xiaoduxiong_epoch_12 --save_dir=./inference_model --fixed_input_shape=[640,960]
```

## 预测部署
> 点击下载测试图片 [xiaoduxiong_test_image.tar.gz](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_test_image.tar.gz)

```
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
result = predictor.predict(image='xiaoduxiong_test_image/JPEGImages/WeChatIMG110.jpeg')
```

## 预测性能对比
### 测试环境

- CUDA 9.0
- CUDNN 7.5
- PaddlePaddle 1.71
- GPU: Tesla P40
- AnalysisPredictor 指采用Python的高性能预测方式
- Executor 指采用paddlepaddle普通的python预测方式
- Batch Size均为1，耗时单位为ms/image，只计算模型运行时间，不包括数据的预处理和后处理

### 性能对比


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
