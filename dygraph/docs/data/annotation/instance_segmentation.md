# 实例分割

实例分割数据的标注推荐使用LabelMe标注工具，如您先前并无安装，那么LabelMe的安装可参考[LabelMe安装和启动](labelme.md)

**注意：LabelMe对于中文支持不够友好，因此请不要在如下的路径以及文件名中出现中文字符!**

## 准备工作  

1. 将收集的图像存放于`JPEGImages`文件夹下，例如存储在`D:\MyDataset\JPEGImages`
2. 创建与图像文件夹相对应的文件夹`Annotations`，用于存储标注的json文件，如`D:MyDataset\Annotations`
3. 打开LabelMe，点击”Open Dir“按钮，选择需要标注的图像所在的文件夹打开，则”File List“对话框中会显示所有图像所对应的绝对路径，接着便可以开始遍历每张图像，进行标注工作  

## 目标边缘标注  

1. 打开多边形标注工具（右键菜单->Create Polygon)以打点的方式圈出目标的轮廓，并在弹出的对话框中写明对应label（当label已存在时点击即可，此处请注意label勿使用中文），具体如下提所示，当框标注错误时，可点击左侧的“Edit Polygons”再点击标注框，通过拖拉进行修改，也可再点击“Delete Polygon”进行删除。
![](./pics/detection2.png)

2. 点击右侧”Save“，将标注结果保存到中创建的文件夹Annotations目录中

## 格式转换

LabelMe标注后的数据还需要进行转换为MSCOCO格式，才可以用于实例分割任务的训练，创建保存目录`D:\dataset_seg`，在python环境中安装paddlex后，使用如下命令即可
```
paddlex --data_conversion --source labelme --to MSCOCO \
        --pics D:\MyDataset\JPEGImages \
        --annotations D:\MyDataset\Annotations \
        --save_dir D:\dataset_coco
```

## 数据集划分

参考文档[数据划分](../split.md)完成训练集和验证集的划分，用于模型训练和精度验证。
