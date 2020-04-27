# 训练图像分类模型

---
本文档训练代码可参考PaddleX的[代码tutorial/train/classification/mobilenetv2.py](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/classification/mobilenetv2.py)

**1.下载并解压训练所需的数据集**

> 使用1张显卡训练并指定使用0号卡。

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx
```

> 这里使用蔬菜数据集，训练集、验证集和测试集共包含6189个样本，18个类别。

```python
veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')
```

**2.定义训练和验证过程中的数据处理和增强操作**
> transforms用于指定训练和验证过程中的数据处理和增强操作流程，如下代码在训练过程中使用了`RandomCrop`和`RandomHorizontalFlip`进行数据增强，transforms的使用见[paddlex.cls.transforms](../../apis/transforms/cls_transforms.html#paddlex-cls-transforms)

```python
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])
```

**3.创建数据集读取器，并绑定相应的数据预处理流程**
> 通过不同的数据集读取器可以加载不同格式的数据集，数据集API的介绍见文档[paddlex.datasets](../../apis/datasets.md)

```python
train_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/train_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=eval_transforms)
```

**4.创建模型进行训练**
> 模型训练会默认自动下载和使用imagenet图像数据集上的预训练模型，用户也可自行指定`pretrain_weights`参数来设置预训练权重。模型训练过程每间隔`save_interval_epochs`轮会保存一次模型在`save_dir`目录下，同时在保存的过程中也会在验证数据集上计算相关指标。

> 分类模型的接口可见文档[paddlex.cls.models](../../apis/models.md)

```python
model = pdx.cls.MobileNetV2(num_classes=len(train_dataset.labels))
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.025,
    save_dir='output/mobilenetv2',
    use_vdl=True)
```

> 将`use_vdl`设置为`True`时可使用VisualDL查看训练指标。按以下方式启动VisualDL后，浏览器打开 https://0.0.0.0:8001即可。其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP。

```shell
visualdl --logdir output/mobilenetv2/vdl_log --port 8001
```

**5.验证或测试**
> 利用训练完的模型可继续在验证集上进行验证。

```python
eval_metrics = model.evaluate(eval_dataset, batch_size=8)
print("eval_metrics:", eval_metrics)
```

> 结果输出：
```
eval_metrics: OrderedDict([('acc1', 0.9895916733386709), ('acc5', 0.9983987189751802)])
```

> 训练完用模型对图片进行测试。

```python
predict_result = model.predict('./vegetables_cls/bocai/IMG_00000839.jpg', topk=5)
print("predict_result:", predict_result)
```

> 结果输出：
```
predict_result: [{'category_id': 13, 'category': 'bocai', 'score': 0.8607276},
                 {'category_id': 11, 'category': 'kongxincai', 'score': 0.06386806},
                 {'category_id': 2, 'category': 'suanmiao', 'score': 0.03736042},
                 {'category_id': 12, 'category': 'heiqiezi', 'score': 0.007879922},
                 {'category_id': 17, 'category': 'huluobo', 'score': 0.006327283}]
```
