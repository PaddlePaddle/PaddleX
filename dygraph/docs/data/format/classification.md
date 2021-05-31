# 图像分类ImageNet

## 目录

* [数据格式](#1)
  * [数据文件夹结构](#11)
  * [训练集、验证集列表和类别标签列表](#12)
* [数据加载](#2)

## <h2 id="1">数据格式</h2>

在PaddleX中，图像分类任务支持的ImageNet数据集格式要求如下：

### <h3 id="11">1. 数据文件夹结构</h3>

数据集目录`data_dir`下包含多个文件夹，每个文件夹中的图像均属于同一个类别，文件夹的命名即为类别名（**注意路径中不要包括中文，空格**）。

文件夹结构示例如下：

```
MyDataset/ # 图像分类数据集根目录
|--dog/ # 当前文件夹所有图片属于dog类别
|  |--d1.jpg
|  |--d2.jpg
|  |--...
|  |--...
|
|--...
|
|--snake/ # 当前文件夹所有图片属于snake类别
|  |--s1.jpg
|  |--s2.jpg
|  |--...
|  |--...
```

### <h3 id="12">2. 训练集、验证集列表和类别标签列表</h3>

**为了完成模型的训练和精度验证。我们需要在`MyDataset`目录下准备`train_list.txt`, `val_list.txt`和`labels.txt`三个文件**，分别用于表示训练集列表，验证集列表和类别标签列表。点击下载[图像分类示例数据集](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz)查看具体的数据格式。


* **labels.txt**  

labels.txt用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数)，例如labels.txt为以下内容
```
dog
cat
snake
```
即表示该分类数据集中共有3个类别，分别为`dog`，`cat`和`snake`，在模型训练中`dog`对应的类别id为0, `cat`对应1，以此类推

* **train_list.txt**  

train_list.txt列出用于训练时的图片集合，与其对应的类别id，示例如下
```
dog/d1.jpg 0
dog/d2.jpg 0
cat/c1.jpg 1
... ...
snake/s1.jpg 2
```
其中第一列为相对对`MyDataset`的相对路径，第二列为图片对应类别的类别id

* **val_list.txt**  

val_list列出用于验证时的图片集成，与其对应的类别id，格式与train_list.txt一致

## <h2 id="2">数据集加载</h2>

训练过程中，PaddleX加载数据集的示例代码如下:

```
import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
     T.RandomCrop(crop_size=224),
     T.RandomHorizontalFlip(),
     T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256),
    T.CenterCrop(crop_size=224),
    T.Normalize()
])

train_dataset = pdx.datasets.ImageNet(
                    data_dir='./MyDataset',
                    file_list='./MyDataset/train_list.txt',
                    label_list='./MyDataset/labels.txt',
                    transforms=train_transforms)
eval_dataset = pdx.datasets.ImageNet(
                    data_dir='./MyDataset',
                    file_list='./MyDataset/eval_list.txt',
                    label_list='./MyDataset/labels.txt',
                    transforms=eval_transforms)
```
