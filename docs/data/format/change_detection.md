# 地块检测ChangeDet

## 数据集文件夹结构

在PaddleX中，**标注文件为png文件**。建议用户将数据集按照如下方式进行组织，同一地块不同时期的地貌原图均放在同一目录，如`JPEGImages`，标注的同名png文件均放在同一目录，如`Annotations`，示例如下
```
MyDataset/ # 语义分割数据集根目录
|--JPEGImages/ # 原图文件所在目录，包含同一物体前期和后期的图片
|  |--1_1.jpg
|  |--1_2.jpg
|  |--2_1.jpg
|  |--2_2.jpg
|  |--...
|  |--...
|
|--Annotations/ # 标注文件所在目录
|  |--1.png
|  |--2.png
|  |--...
|  |--...
```
同一地块不同时期的地貌原图，如1_1.jpg和1_2.jpg，可以是RGB彩色图像、灰度图、或tiff格式的多通道图像。语义分割的标注图像，如1.png，为单通道图像，像素标注类别需要从0开始递增（一般0表示background背景), 例如0， 1， 2， 3表示4种类别，标注类别最多255个类别(其中像素值255不参与训练和评估)。

## 划分训练集验证集

**为了用于训练，我们需要在`MyDataset`目录下准备`train_list.txt`, `val_list.txt`和`labels.txt`三个文件**，分别用于表示训练集列表，验证集列表和类别标签列表。

**labels.txt**  

labels.txt用于列出所有类别，类别对应行号表示模型训练过程中类别的id(行号从0开始计数)，例如labels.txt为以下内容
```
unchanged
changed
```
表示该检测数据集中共有2个分割类别，分别为`unchanged`和`changed`，在模型训练中`unchanged`对应的类别id为0, `changed`对应1，以此类推，如不知具体类别标签，可直接在labels.txt逐行写0，1，2...序列即可。

**train_list.txt**  

train_list.txt列出用于训练时的图片集合，与其对应的标注文件，示例如下
```
JPEGImages/1_1.jpg JPEGImages/1_2.jpg Annotations/1.png
JPEGImages/2_1.jpg JPEGImages/2_2.jpg Annotations/2.png
... ...
```
其中第一列和第二列为原图相对`MyDataset`的相对路径，对应同一地块不同时期的地貌图像，第三列为标注文件相对`MyDataset`的相对路径

**val_list.txt**  

val_list列出用于验证时的图片集成，与其对应的标注文件，格式与val_list.txt一致

## PaddleX数据集加载  

[示例代码](https://github.com/PaddlePaddle/PaddleX/tree/release/1.3/examples/change_detection/train.py)
