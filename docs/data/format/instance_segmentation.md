# 实例分割MSCOCO

## 目录

* [数据格式](#1)
  * [数据文件夹结构](#11)
  * [训练集、验证集列表](#12)
* [数据加载](#2)
* [添加负样本](#3)


## <h2 id="1">数据格式</h2>

在PaddleX中，实例分割支持MSCOCO数据集格式。（MSCOCO格式同样也可以用于目标检测）。

### <h3 id="11">数据文件夹结构</h2>

数据集按照如下方式进行组织，原图均放在同一目录，如JPEGImages，标注文件（如annotations.json）放在与JPEGImages所在目录同级目录下，示例结构如下
```
MyDataset/ # 实例分割数据集根目录
|--JPEGImages/ # 原图文件所在目录
|  |--1.jpg
|  |--2.jpg
|  |--...
|  |--...
|
|--annotations.json # 标注文件所在目录
```

## <h3 id="12">训练集、验证集列表</h3>

为了区分训练集和验证集，在`MyDataset`同级目录，使用不同的json表示数据的划分，例如`train.json`和`val.json`。点击下载[实例分割示例数据集](https://bj.bcebos.com/paddlex/datasets/garbage_ins_det.tar.gz)查看具体的数据格式。


## <h2 id="2">数据集加载</h2>

训练过程中，PaddleX加载数据集的示例代码如下:

```python
import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
    T.RandomResizeByShort(
        short_sizes=[640, 672, 704, 736, 768, 800],
        max_size=1333,
        interp='CUBIC'),
    T.RandomHorizontalFlip(),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.ResizeByShort(
        short_size=800, max_size=1333, interp='CUBIC'),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = pdx.dataset.CocoDetection(
                    data_dir='./MyDataset/JPEGImages',
                    ann_file='./MyDataset/train.json',
                    transforms=train_transforms)
eval_dataset = pdx.dataset.CocoDetection(
                    data_dir='./MyDataset/JPEGImages',
                    ann_file='./MyDataset/val.json',
                    transforms=eval_transforms)
```


## <h2 id="3">添加负样本</h2>

实例分割任务支持添加负样本进行训练以降低误检率，代码示例如下：

```python
import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
    T.RandomResizeByShort(
        short_sizes=[640, 672, 704, 736, 768, 800],
        max_size=1333,
        interp='CUBIC'),
    T.RandomHorizontalFlip(),
    T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 情况一：ann_file中已经包含负样本
# 要求每个负样本都有对应的标注数据，标注数据里没有标注框
train_dataset = pdx.dataset.CocoDetection(
                    data_dir='./MyDataset/JPEGImages',
                    ann_file='./MyDataset/train.json',
                    transforms=train_transforms,
                    allow_empty=True,   # 是否加载负样本
                    empty_ratio=1.)   # 用于指定负样本占总样本数的比例。如果小于0或大于等于1，则保留全部的负样本。默认为1。

# 情况二：train_list中仅包含正样本，负样本在单独的路径下
# 不要求负样本有标注数据
train_dataset = pdx.dataset.CocoDetection(
                    data_dir='./MyDataset/JPEGImages',
                    ann_file='./MyDataset/train.json',
                    transforms=train_transforms)
train_dataset.add_negative_samples(
                        image_dir='path/to/negative/images',   # 背景图片所在的文件夹目录。
                        empty_ratio=1)   # 用于指定负样本占总样本数的比例。如果为None，保留数据集初始化是设置的`empty_ratio`值，
                                         # 否则更新原有`empty_ratio`值。如果小于0或大于等于1，则保留全部的负样本。默认为1。

```
