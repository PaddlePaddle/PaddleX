# 语义分割Seg

## 数据集文件夹结构

在PaddleX中，**标注文件为png文件**。建议用户将数据集按照如下方式进行组织，原图均放在同一目录，如`JPEGImages`，标注的同名png文件均放在同一目录，如`Annotations`，示例如下
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

## 划分训练集验证集

**为了用于训练，我们需要在`MyDataset`目录下准备`train_list.txt`, `val_list.txt`和`labels.txt`三个文件**，分别用于表示训练集列表，验证集列表和类别标签列表。[点击下载语义分割示例数据集](https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz)

<!--
> 注：也可使用PaddleX自带工具，对数据集进行随机划分，**在数据集按照上面格式组织后**，使用如下命令即可快速完成数据集随机划分，其中split指标训练集的比例，剩余的比例用于验证集。
> ```
> paddlex --split_dataset --from Seg --pics ./JPEGImages --annotations ./Annotations --split 0.8 --save_dir ./splited_dataset_dir
> ```
-->

**labels.txt**  

labels.txt用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数)，例如labels.txt为以下内容
```
backgrond
human
car
```
表示该检测数据集中共有3个分割类别，分别为`background`，`human`和`car`，在模型训练中`background`对应的类别id为0, `human`对应1，以此类推，如不知具体类别标签，可直接在labels.txt逐行写0，1，2...序列即可。

**train_list.txt**  

train_list.txt列出用于训练时的图片集合，与其对应的标注文件，示例如下
```
JPEGImages/1.jpg Annotations/1.png
JPEGImages/2.jpg Annotations/2.png
... ...
```
其中第一列为原图相对`MyDataset`的相对路径，第二列为标注文件相对`MyDataset`的相对路径

**val_list.txt**  

val_list列出用于验证时的图片集成，与其对应的标注文件，格式与val_list.txt一致

## PaddleX数据集加载  

示例代码如下，
```
import paddlex as pdx
from paddlex.seg import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ResizeRangeScaling(),
    transforms.RandomPaddingCrop(crop_size=512),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.ResizeByLong(long_size=512),
    transforms.Padding(target_size=512),
    transforms.Normalize()
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
