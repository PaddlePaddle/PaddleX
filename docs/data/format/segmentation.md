# 语义分割Seg

## 目录

* [数据格式](#1)
  * [数据文件夹结构](#11)
  * [训练集、验证集列表和类别标签列表](#12)
* [数据加载](#2)


## <h2 id="1">数据格式</h2>

### <h3 id="11">数据文件夹结构</h2>

数据集按照如下方式进行组织，原图均放在同一目录，如`JPEGImages`，标注的同名png文件均放在同一目录，如`Annotations`。示例如下：
```
MyDataset/ # 语义分割数据集根目录
|--JPEGImages/ # 原图文件所在目录
|  |--1.jpg
|  |--2.jpg
|  |--...
|  |--...
|
|--Annotations/ # 标注文件所在目录
|  |--1.png
|  |--2.png
|  |--...
|  |--...
```
语义分割的标注图像，如1.png，为单通道图像,像素标注类别需要从0开始递增（一般0表示background背景), 例如0， 1， 2， 3表示4种类别，标注类别最多255个类别(其中像素值255不参与训练和评估)。


### <h3 id="12">训练集、验证集列表和类别标签列表</h3>

**为了用于训练，我们需要在`MyDataset`目录下准备`train_list.txt`, `val_list.txt`和`labels.txt`三个文件**，分别用于表示训练集列表，验证集列表和类别标签列表。点击下载[语义分割示例数据集](https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz)查看具体的数据格式。


* **labels.txt**  

labels.txt用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数)，例如labels.txt为以下内容
```
background
human
car
```
表示该检测数据集中共有3个分割类别，分别为`background`，`human`和`car`，在模型训练中`background`对应的类别id为0, `human`对应1，以此类推，如不知具体类别标签，可直接在labels.txt逐行写0，1，2...序列即可。

* **train_list.txt**  

train_list.txt列出用于训练时的图片集合，与其对应的标注文件，示例如下
```
JPEGImages/1.jpg Annotations/1.png
JPEGImages/2.jpg Annotations/2.png
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
    T.Resize(target_size=512),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_dataset = pdx.datasets.SegDataset(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/train_list.txt',
                        label_list='./MyDataset/labels.txt',
                        transforms=train_transforms)
eval_dataset = pdx.datasets.SegDataset(
                        data_dir='./MyDataset',
                        file_list='./MyDataset/val_list.txt',
                        label_list='MyDataset/labels.txt',
                        transforms=eval_transforms)
```
