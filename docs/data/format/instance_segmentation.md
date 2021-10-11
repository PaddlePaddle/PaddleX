# 实例分割MSCOCO

## 目录

* [数据格式](#1)
  * [数据文件夹结构](#11)
  * [训练集、验证集列表](#12)
* [数据加载](#2)


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
