# Python部署
PaddleX已经集成了基于Python的高性能预测接口，在安装PaddleX后，可参照如下代码示例，进行预测。相关的接口文档可参考[paddlex.deploy](../../../apis/deploy.md)

## 导出预测模型
可参考[模型导出](../export_model.md)将模型导出为inference格式的模型。

## 预测部署
> 点击下载测试图片 [xiaoduxiong_test_image.tar.gz](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_test_image.tar.gz)

```
import paddlex as pdx
predictor = pdx.deploy.Predictor('./inference_model')
result = predictor.predict(image='xiaoduxiong_test_image/JPEGImages/WeChatIMG110.jpeg')
```

> 关于预测速度的说明：采用Paddle的Predictor进行预测时，由于涉及到内存显存初始化等原因，在模型加载后刚开始预测速度会较慢，一般在模型运行20~50后（即预测20~30张图片）预测速度才会稳定。

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
