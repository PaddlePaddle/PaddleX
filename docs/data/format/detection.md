# 目标检测PascalVOC

## 目录

* [数据格式](#1)
  * [数据文件夹结构](#11)
  * [训练集、验证集列表和类别标签列表](#12)
* [数据加载](#2)
* [添加负样本](#3)


## <h2 id="1">数据格式</h2>

在PaddleX中，目标检测支持PascalVOC数据集格式。

### <h3 id="11">数据文件夹结构</h2>

数据集按照如下方式进行组织，原图均放在同一目录，如`JPEGImages`，标注的同名xml文件均放在同一目录，如`Annotations`，示例如下：

```
MyDataset/ # 目标检测数据集根目录
|--JPEGImages/ # 原图文件所在目录
|  |--1.jpg
|  |--2.jpg
|  |--...
|  |--...
|
|--Annotations/ # 标注文件所在目录
|  |--1.xml
|  |--2.xml
|  |--...
|  |--...
```

### <h3 id="12">训练集、验证集列表和类别标签列表</h3>

**为了用于训练，我们需要在`MyDataset`目录下准备`train_list.txt`, `val_list.txt`和`labels.txt`三个文件**，分别用于表示训练集列表，验证集列表和类别标签列表。点击下载[目标检测示例数据集](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)查看具体的数据格式。

* **labels.txt**  

labels.txt用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数)，例如labels.txt为以下内容
```
dog
cat
snake
```
表示该检测数据集中共有3个目标类别，分别为`dog`，`cat`和`snake`，在模型训练中`dog`对应的类别id为0, `cat`对应1，以此类推

* **train_list.txt**  

train_list.txt列出用于训练时的图片集合，与其对应的标注文件，示例如下
```
JPEGImages/1.jpg Annotations/1.xml
JPEGImages/2.jpg Annotations/2.xml
... ...
```
其中第一列为原图相对`MyDataset`的相对路径，第二列为标注文件相对`MyDataset`的相对路径

* **val_list.txt**  

val_list列出用于验证时的图片集成，与其对应的标注文件，格式与val_list.txt一致


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


train_dataset = pdx.datasets.VOCDetection(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/train_list.txt',
                        label_list='./MyDataset/labels.txt',
                        transforms=train_transforms)
eval_dataset = pdx.datasets.VOCDetection(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/val_list.txt',
                        label_list='MyDataset/labels.txt',
                        transforms=eval_transforms)
```

## <h2 id="3">添加负样本</h2>

检测任务支持添加负样本进行训练以降低误检率，代码示例如下：

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

# 情况一：train_list中已经包含负样本
train_dataset = pdx.datasets.VOCDetection(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/train_list.txt',
                        label_list='./MyDataset/labels.txt',
                        transforms=train_transforms,
                        allow_empty=True,   # 是否加载负样本
                        empty_ratio=1.)   # 用于指定负样本占总样本数的比例。如果小于0或大于等于1，则保留全部的负样本。默认为1。

# 情况二：train_list中仅包含正样本，负样本在单独的路径下
train_dataset = pdx.datasets.VOCDetection(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/train_list.txt',
                        label_list='./MyDataset/labels.txt',
                        transforms=train_transforms)
train_dataset.add_negative_samples(
                        image_dir='path/to/negative/images',   # 背景图片所在的文件夹目录。
                        empty_ratio=1)   # 用于指定负样本占总样本数的比例。如果为None，保留数据集初始化是设置的`empty_ratio`值，
                                         # 否则更新原有`empty_ratio`值。如果小于0或大于等于1，则保留全部的负样本。默认为1。

```
