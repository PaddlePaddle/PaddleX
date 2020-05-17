# 数据集格式说明

---
## 图像分类ImageNet

图像分类ImageNet数据集包含对应多个标签的图像文件夹、标签文件及图像列表文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录
|--labelA  # 标签为labelA的图像目录
|  |--a1.jpg
|  |--...
|  └--...
|
|--...
|
|--labelZ  # 标签为labelZ的图像目录
|  |--z1.jpg
|  |--...
|  └--...
|
|--train_list.txt  # 训练文件列表文件
|
|--val_list.txt  # 验证文件列表文件
|
└--labels.txt  # 标签列表文件

```
其中，相应的文件名可根据需要自行定义。

`train_list.txt`和`val_list.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为图像文件对应的标签id(从0开始)。如下所示：
```
labelA/a1.jpg 0
labelZ/z1.jpg 25
...
```

`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```
[点击这里](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz)，下载蔬菜分类分类数据集。  
在PaddleX中，使用`paddlex.cv.datasets.ImageNet`([API说明](./apis/datasets.html#imagenet))加载分类数据集。

## 目标检测VOC
目标检测VOC数据集包含图像文件夹、标注信息文件夹、标签文件及图像列表文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录
|--JPEGImages  # 图像目录
|  |--xxx1.jpg
|  |--...
|  └--...
|
|--Annotations  # 标注信息目录
|  |--xxx1.xml
|  |--...
|  └--...
|
|--train_list.txt  # 训练文件列表文件
|
|--val_list.txt  # 验证文件列表文件
|
└--labels.txt  # 标签列表文件

```
其中，相应的文件名可根据需要自行定义。

`train_list.txt`和`val_list.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为标注文件相对于dataset的相对路径。如下所示：
```
JPEGImages/xxx1.jpg Annotations/xxx1.xml
JPEGImages/xxx2.jpg Annotations/xxx2.xml
...
```

`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```
[点击这里](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)，下载昆虫检测数据集。  
在PaddleX中，使用`paddlex.cv.datasets.VOCDetection`([API说明](./apis/datasets.html#vocdetection))加载目标检测VOC数据集。

## 目标检测和实例分割COCO
目标检测和实例分割COCO数据集包含图像文件夹及图像标注信息文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录
|--JPEGImages  # 图像目录
|  |--xxx1.jpg
|  |--...
|  └--...
|
|--train.json  # 训练相关信息文件
|
└--val.json  # 验证相关信息文件

```
其中，相应的文件名可根据需要自行定义。

`train.json`和`val.json`存储与标注信息、图像文件相关的信息。如下所示：

```
{
  "annotations": [
    {
      "iscrowd": 0,
      "category_id": 1,
      "id": 1,
      "area": 33672.0,
      "image_id": 1,
      "bbox": [232, 32, 138, 244],
      "segmentation": [[32, 168, 365, 117, ...]]
    },
    ...
  ],
  "images": [
    {
      "file_name": "xxx1.jpg",
      "height": 512,
      "id": 267,
      "width": 612
    },
    ...
  ]
  "categories": [
    {
      "name": "labelA",
      "id": 1,
      "supercategory": "component"
    }
  ]
}
```
其中，每个字段的含义如下所示：

| 域名 | 字段名 | 含义 | 数据类型 | 备注 |
|:-----|:--------|:------------|------|:-----|
| annotations | id | 标注信息id | int | 从1开始 |
| annotations | iscrowd      | 标注框是否为一组对象 | int | 只有0、1两种取值 |
| annotations | category_id  | 标注框类别id | int |  |
| annotations | area         | 标注框的面积 | float |  |
| annotations | image_id     | 当前标注信息所在图像的id | int |  |
| annotations | bbox         | 标注框坐标 | list | 长度为4，分别代表x,y,w,h |
| annotations | segmentation | 标注区域坐标 | list | list中有至少1个list，每个list由每个小区域坐标点的横纵坐标(x,y)组成 |
| images          | id                | 图像id | int | 从1开始 |
| images   | file_name         | 图像文件名 | str |  |
| images      | height            | 图像高度 | int |  |
| images       | width             | 图像宽度 | int |  |
| categories  | id            | 类别id | int | 从1开始 |
| categories | name          | 类别标签名 | str |  |
| categories | supercategory | 类别父类的标签名 | str |  |


[点击这里](https://bj.bcebos.com/paddlex/datasets/garbage_ins_det.tar.gz)，下载垃圾实例分割数据集。  
在PaddleX中，使用`paddlex.cv.datasets.COCODetection`([API说明](./apis/datasets.html#cocodetection))加载COCO格式数据集。

## 语义分割数据
语义分割数据集包含原图、标注图及相应的文件列表文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录
|--images  # 原图目录
|  |--xxx1.png
|  |--...
|  └--...
|
|--annotations  # 标注图目录
|  |--xxx1.png
|  |--...
|  └--...
|
|--train_list.txt  # 训练文件列表文件
|
|--val_list.txt  # 验证文件列表文件
|
└--labels.txt  # 标签列表

```
其中，相应的文件名可根据需要自行定义。

`train_list.txt`和`val_list.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为标注图像文件相对于dataset的相对路径。如下所示：
```
images/xxx1.png annotations/xxx1.png
images/xxx2.png annotations/xxx2.png
...
```

`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
background
labelA
labelB
...
```

标注图像为单通道图像，像素值即为对应的类别,像素标注类别需要从0开始递增（一般第一个类别为`background`），
例如0，1，2，3表示有4种类别，标注类别最多为256类。其中可以指定特定的像素值用于表示该值的像素不参与训练和评估（默认为255）。

[点击这里](https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz)，下载视盘语义分割数据集。  
在PaddleX中，使用`paddlex.cv.datasets.SegReader`([API说明](./apis/datasets.html#segreader))加载语义分割数据集。


## 图像分类EasyDataCls

图像分类EasyDataCls数据集包含存放图像和json文件的文件夹、标签文件及图像列表文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录
|--easydata  # 存放图像和json文件的文件夹
|  |--0001.jpg
|  |--0001.json
|  |--0002.jpg
|  |--0002.json
|  └--...
|
|--train_list.txt  # 训练文件列表文件
|
|--val_list.txt  # 验证文件列表文件
|
└--labels.txt  # 标签列表文件

```
其中，图像文件名应与json文件名一一对应。   

每个json文件存储于`labels`相关的信息。如下所示：
```
{"labels": [{"name": "labelA"}]}
```
其中，`name`字段代表对应图像的类别。  

`train_list.txt`和`val_list.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为json文件相对于dataset的相对路径。如下所示：
```
easydata/0001.jpg easydata/0001.json
easydata/0002.jpg easydata/0002.json
...
```

`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```
[点击这里](https://ai.baidu.com/easydata/)，可以标注图像分类EasyDataCls数据集。  
在PaddleX中，使用`paddlex.cv.datasets.EasyDataCls`([API说明](./apis/datasets.html#easydatacls))加载分类数据集。


## 目标检测和实例分割EasyDataDet

目标检测和实例分割EasyDataDet数据集包含存放图像和json文件的文件夹、标签文件及图像列表文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录ß
|--easydata  # 存放图像和json文件的文件夹
|  |--0001.jpg
|  |--0001.json
|  |--0002.jpg
|  |--0002.json
|  └--...
|
|--train_list.txt  # 训练文件列表文件
|
|--val_list.txt  # 验证文件列表文件
|
└--labels.txt  # 标签列表文件

```
其中，图像文件名应与json文件名一一对应。   

每个json文件存储于`labels`相关的信息。如下所示：
```
"labels": [{"y1": 18, "x2": 883, "x1": 371, "y2": 404, "name": "labelA", 
            "mask": "kVfc0`0Zg0<F7J7I5L5K4L4L4L3N3L3N3L3N2N3M2N2N2N2N2N2N1O2N2O1N2N1O2O1N101N1O2O1N101N10001N101N10001N10001O0O10001O000O100000001O0000000000000000000000O1000001O00000O101O000O101O0O101O0O2O0O101O0O2O0O2N2O0O2O0O2N2O1N1O2N2N2O1N2N2N2N2N2N2M3N3M2M4M2M4M3L4L4L4K6K5J7H9E\\iY1"}, 
           {"y1": 314, "x2": 666, "x1": 227, "y2": 676, "name": "labelB",
            "mask": "mdQ8g0Tg0:G8I6K5J5L4L4L4L4M2M4M2M4M2N2N2N3L3N2N2N2N2O1N1O2N2N2O1N1O2N2O0O2O1N1O2O0O2O0O2O001N100O2O000O2O000O2O00000O2O000000001N100000000000000000000000000000000001O0O100000001O0O10001N10001O0O101N10001N101N101N101N101N2O0O2N2O0O2N2N2O0O2N2N2N2N2N2N2N2N2N3L3N2N3L3N3L4M2M4L4L5J5L5J7H8H;BUcd<"}, 
           ...]}
```
其中，list中的每个元素代表一个标注信息，标注信息中字段的含义如下所示： 

| 字段名 | 含义 | 数据类型 | 备注 |
|:--------|:------------|------|:-----|
| x1 | 标注框左下角横坐标 | int | |
| y1 | 标注框左下角纵坐标 | int | |
| x2 | 标注框右上角横坐标 | int | |
| y2 | 标注框右上角纵坐标 | int | |
| name | 标注框中物体类标 | str | |
| mask | 分割区域布尔型numpy编码后的字符串 | str | 该字段可以不存在，当不存在时只能进行目标检测 |

`train_list.txt`和`val_list.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为json文件相对于dataset的相对路径。如下所示：
```
easydata/0001.jpg easydata/0001.json
easydata/0002.jpg easydata/0002.json
...
```

`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```

[点击这里](https://ai.baidu.com/easydata/)，可以标注图像分类EasyDataDet数据集。  
在PaddleX中，使用`paddlex.cv.datasets.EasyDataDet`([API说明](./apis/datasets.html#easydatadet))加载分类数据集。

## 语义分割EasyDataSeg

语义分割EasyDataSeg数据集包含存放图像和json文件的文件夹、标签文件及图像列表文件。
参考数据文件结构如下：
```
./dataset/  # 数据集根目录ß
|--easydata  # 存放图像和json文件的文件夹
|  |--0001.jpg
|  |--0001.json
|  |--0002.jpg
|  |--0002.json
|  └--...
|
|--train_list.txt  # 训练文件列表文件
|
|--val_list.txt  # 验证文件列表文件
|
└--labels.txt  # 标签列表文件

```
其中，图像文件名应与json文件名一一对应。   

每个json文件存储于`labels`相关的信息。如下所示：
```
"labels": [{"y1": 18, "x2": 883, "x1": 371, "y2": 404, "name": "labelA", 
            "mask": "kVfc0`0Zg0<F7J7I5L5K4L4L4L3N3L3N3L3N2N3M2N2N2N2N2N2N1O2N2O1N2N1O2O1N101N1O2O1N101N10001N101N10001N10001O0O10001O000O100000001O0000000000000000000000O1000001O00000O101O000O101O0O101O0O2O0O101O0O2O0O2N2O0O2O0O2N2O1N1O2N2N2O1N2N2N2N2N2N2M3N3M2M4M2M4M3L4L4L4K6K5J7H9E\\iY1"}, 
           {"y1": 314, "x2": 666, "x1": 227, "y2": 676, "name": "labelB",
            "mask": "mdQ8g0Tg0:G8I6K5J5L4L4L4L4M2M4M2M4M2N2N2N3L3N2N2N2N2O1N1O2N2N2O1N1O2N2O0O2O1N1O2O0O2O0O2O001N100O2O000O2O000O2O00000O2O000000001N100000000000000000000000000000000001O0O100000001O0O10001N10001O0O101N10001N101N101N101N101N2O0O2N2O0O2N2N2O0O2N2N2N2N2N2N2N2N2N3L3N2N3L3N3L4M2M4L4L5J5L5J7H8H;BUcd<"}, 
           ...]}
```
其中，list中的每个元素代表一个标注信息，标注信息中字段的含义如下所示： 

| 字段名 | 含义 | 数据类型 | 备注 |
|:--------|:------------|------|:-----|
| x1 | 标注框左下角横坐标 | int | |
| y1 | 标注框左下角纵坐标 | int | |
| x2 | 标注框右上角横坐标 | int | |
| y2 | 标注框右上角纵坐标 | int | |
| name | 标注框中物体类标 | str | |
| mask | 分割区域布尔型numpy编码后的字符串 | str | 该字段必须存在 |

`train_list.txt`和`val_list.txt`文本以空格为分割符分为两列，第一列为图像文件相对于dataset的相对路径，第二列为json文件相对于dataset的相对路径。如下所示：
```
easydata/0001.jpg easydata/0001.json
easydata/0002.jpg easydata/0002.json
...
```

`labels.txt`: 每一行为一个单独的类别，相应的行号即为类别对应的id（行号从0开始)，如下所示：
```
labelA
labelB
...
```

[点击这里](https://ai.baidu.com/easydata/)，可以标注图像分类EasyDataSeg数据集。  
在PaddleX中，使用`paddlex.cv.datasets.EasyDataSeg`([API说明](./apis/datasets.html#easydataseg))加载分类数据集。