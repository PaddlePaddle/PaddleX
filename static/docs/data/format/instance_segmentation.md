# 实例分割MSCOCO

## 数据集文件夹结构

在PaddleX中，实例分割支持MSCOCO数据集格式（MSCOCO格式同样也可以用于目标检测）。建议用户将数据集按照如下方式进行组织，原图均放在同一目录，如JPEGImages，标注文件（如annotations.json）放在与JPEGImages所在目录同级目录下，示例结构如下
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

## 划分训练集验证集

在PaddleX中，为了区分训练集和验证集，在`MyDataset`同级目录，使用不同的json表示数据的划分，例如`train.json`和`val.json`。[点击下载实例分割示例数据集](https://bj.bcebos.com/paddlex/datasets/garbage_ins_det.tar.gz)。

> 注：也可使用PaddleX自带工具，对数据集进行随机划分，**在数据集按照上面格式组织后**，使用如下命令即可快速完成数据集随机划分，其中val_value表示验证集的比例，test_value表示测试集的比例（可以为0），剩余的比例用于训练集。
> ```
> paddlex --split_dataset --format COCO --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
> ```

MSCOCO数据的标注文件采用json格式，用户可使用Labelme, 精灵标注助手或EasyData等标注工具进行标注，参见[数据标注工具](../annotation.md)

## PaddleX加载数据集
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

train_dataset = pdx.datasets.CocoDetection(
                    data_dir='./MyDataset/JPEGImages',
                    ann_file='./MyDataset/train.json',
                    transforms=train_transforms)
eval_dataset = pdx.datasets.CocoDetection(
                    data_dir='./MyDataset/JPEGImages',
                    ann_file='./MyDataset/val.json',
                    transforms=eval_transforms)
```
