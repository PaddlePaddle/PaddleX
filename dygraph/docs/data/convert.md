# 数据格式转换

PaddleX支持图像分类、目标检测、实例分割和语义分割四大视觉领域常见的任务，对于每类视觉任务，都支持了特定的数据格式。PaddleX目前支持了图像分类的ImageNet格式，目标检测的PascalVOC格式，实例分割的MSCOCO格式（MSCOCO也可以用于目标检测）以及语义分割数据格式。

## 常见标注工具

 图像分类无需标注工具，用户只需以txt文件记录每张图片的类别标签即可。对于目标检测、实例分割和语义分割，PaddleX已经与主流的标注工具进行了适配，用户可根据自己的需求，选择以下标注工具进行数据标注。

| 标注工具    | 图像分类 | 目标检测 | 实例分割 | 语义分割 | 安装                                             |
| :---------  | :------- | :------ | :------  | :------- | :----------------------------------------------- |
| Labelme     | -        | √        | √        | √        | pip install labelme （本地数据标注）                              |
| 精灵标注    | √        | √*        | √        | √        | [官网下载](http://www.jinglingbiaozhu.com/) （本地数据标注）     |

数据标注完成后，参照如下流程，将标注数据转为可用PaddleX模型训练的数据组织格式。

***注意**：精灵标注的目标检测数据可以在工具内部导出为PascalVOC格式，因此paddlex未提供精灵标注数据到PascalVOC格式的转换


## 标注数据格式转换

目前所有标注工具生成的标注文件，均为与原图同名的json格式文件，如`1.jpg`在标注完成后，则会在标注文件保存的目录中生成`1.json`文件。转换时参照以下步骤：

1. 将所有的原图文件放在同一个目录下，如`pics`目录  
2. 将所有的标注json文件放在同一个目录下，如`annotations`目录  
3. 使用如下命令进行转换:

```
paddlex --data_conversion --source labelme --to PascalVOC --pics ./pics --annotations ./annotations --save_dir ./converted_dataset_dir
```

| 参数 | 说明 |
| ---- | ---- |
| --source | 表示数据标注来源，支持`labelme`、`jingling`（分别表示数据来源于LabelMe，精灵标注助手）|
| --to | 表示数据需要转换成为的格式，支持`ImageNet`（图像分类）、`PascalVOC`（目标检测），`MSCOCO`（实例分割，也可用于目标检测）和`SEG`(语义分割)  |
| --pics | 指定原图所在的目录路径  |
| --annotations | 指定标注文件所在的目录路径 |

**注意**：  
1. 精灵标注的目标检测数据可以在工具内部导出为PascalVOC格式，因此paddlex未提供精灵标注数据到PascalVOC格式的转换  
2. 在将LabelMe数据集转换为COCO数据集时，LabelMe的图像文件名和json文件名需要一一对应，才可正确转换
