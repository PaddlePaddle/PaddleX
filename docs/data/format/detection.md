# 目标检测PascalVOC

## 数据集文件夹结构

在PaddleX中，目标检测支持PascalVOC数据集格式。建议用户将数据集按照如下方式进行组织，原图均放在同一目录，如`JPEGImages`，标注的同名xml文件均放在同一目录，如`Annotations`，示例如下
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

## 划分训练集验证集

**为了用于训练，我们需要在`MyDataset`目录下准备`train_list.txt`, `val_list.txt`和`labels.txt`三个文件**，分别用于表示训练集列表，验证集列表和类别标签列表。[点击下载目标检测示例数据集](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)

> 注：也可使用PaddleX自带工具，对数据集进行随机划分，**在数据集按照上面格式组织后**，使用如下命令即可快速完成数据集随机划分，其中val_value表示验证集的比例，test_value表示测试集的比例（可以为0），剩余的比例用于训练集。
> ```
> paddlex --split_dataset --form VOC --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
> ```

**labels.txt**  

labels.txt用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数)，例如labels.txt为以下内容
```
dog
cat
snake
```
表示该检测数据集中共有3个目标类别，分别为`dog`，`cat`和`snake`，在模型训练中`dog`对应的类别id为0, `cat`对应1，以此类推

**train_list.txt**  

train_list.txt列出用于训练时的图片集合，与其对应的标注文件，示例如下
```
JPEGImages/1.jpg Annotations/1.xml
JPEGImages/2.jpg Annotations/2.xml
... ...
```
其中第一列为原图相对`MyDataset`的相对路径，第二列为标注文件相对`MyDataset`的相对路径

**val_list.txt**  

val_list列出用于验证时的图片集成，与其对应的标注文件，格式与val_list.txt一致

## PaddleX数据集加载  
示例代码如下，
```
import paddlex as pdx
from paddlex.det import transforms

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),
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
