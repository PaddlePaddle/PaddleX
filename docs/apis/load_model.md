# 模型加载-load_model

PaddleX提供了统一的模型加载接口，支持加载PaddleX保存的模型，并在验证集上进行评估或对测试图片进行预测

## 函数接口

```
paddlex.load_model(model_dir)
```

### 参数

* **model_dir**: 训练过程中保存的模型路径

### 返回值
* **paddlex.cv.models**, 模型类。

### 示例
> 1. [点击下载](https://bj.bcebos.com/paddlex/models/xiaoduxiong_epoch_12.tar.gz)PaddleX在小度熊分拣数据上训练的MaskRCNN模型
> 2. [点击下载](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_ins_det.tar.gz)小度熊分拣数据集

```
import paddlex as pdx

model_dir = './xiaoduxiong_epoch_12'
data_dir = './xiaoduxiong_ins_det/JPEGImages'
ann_file = './xiaoduxiong_ins_det/val.json'

# 加载垃圾分拣模型
model = pdx.load_model(model_dir)

# 预测
pred_result = model.predict('./xiaoduxiong_ins_det/JPEGImages/WechatIMG114.jpeg')

# 在验证集上进行评估
eval_reader = pdx.cv.datasets.CocoDetection(data_dir=data_dir,
                                            ann_file=ann_file,
                                            transforms=model.eval_transforms)
eval_result = model.evaluate(eval_reader, batch_size=1)
```
