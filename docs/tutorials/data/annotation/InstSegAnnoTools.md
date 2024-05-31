# 通用实例分割数据标注指南

本文档将介绍如何使用 [Labelme](https://github.com/wkentaro/labelme) 标注工具完成实例分割相关单模型的数据标注。
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程，以下提供简洁版本说明：
## 1. 标注数据示例
该数据集是水果实例分割数据集，涵盖五种不同的水果，包含目标不同角度的拍摄照片。
图片示例：
<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/example1.png' width='300px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/example2.png' width='300px'>
<br>
<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/example3.png' width='300px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/example4.png' width='300px'>
</center>

## 2. Labelme 标注工具使用
### 2.1. Labelme 标注工具介绍
Labelme 是一个 python 语言编写，带有图形界面的图像标注软件。可用于图像分类，目标检测，图像分割等任务，在实例分割的标注任务中，标签存储为 JSON 文件。
### 2.2. Labelme 安装
为避免环境冲突，建议在 conda 环境下安装。
```shell
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```
### 2.3. Labelme 的标注过程
#### 2.3.1. 准备待标注数据
1. 创建数据集根目录，如 fruit。
2. 在 fruit 中创建 images 目录（必须为 images 目录），并将待标注图片存储在 images 目录下，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/image_dir.png' width='600px'>
</center>

3. 在 fruit 文件夹中创建待标注数据集的类别标签文件 label.txt，并在 label.txt 中按行写入待标注数据集的类别。以水果实例分割数据集的 label.txt 为例，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/label_txt.png' width='600px'>
</center>

#### 2.3.2. 启动 Labelme
终端进入到带标注数据集根目录，并启动 labelme 标注工具。
```shell
cd path/to/fruit
labelme images --labels label.txt --nodata --autosave --output annotations
```
* --labels 类别标签路径。
* --nodata 停止将图像数据存储到JSON文件。
* --autosave 自动存储。
* --ouput 标签文件存储路径。
#### 2.3.3. 开始图片标注
1. 启动 labelme 后如图所示：


<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/labelme.png' width='600px'>
</center>

2. 点击 "Edit" 选择标注类型，选则 "Create Polygons"。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/edit.png' width='200px'>
</center>

3. 在图片上创建多边形描绘分割区域边界。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/select_target_area.png' width='300px'>
</center>

4. 再次点击选择分割区域类别。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/select_category.png' width='200px'>
</center>

5. 标注好后点击存储。（若在启动 labelme 时未指定 --output 字段，会在第一次存储时提示选择存储路径，若指定 --autosave 字段使用自动保存，则无需点击存储按钮）。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/save.png' width='100px'>
</center>

6. 然后点击 "Next Image" 进行下一张图片的标注。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/next_image.png' width='100px'>
</center>

7. 最终标注好的标签文件如图所示。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/annotation_result.png' width='600px'>
</center>

8. 调整目录得到水果实例分割标准 labelme 格式数据集。

&emsp;&emsp;
在数据集根目录创建 train_anno_list.txt 和 val_anno_list.txt 两个文本文件，并将 annotations 目录下的全部 json 文件路径按一定比例分别写入 train_anno_list.txt 和 val_anno_list.txt，也可全部写入到 train_anno_list.txt 同时创建一个空的 val_anno_list.txt 文件，待上传零代码使用数据划分功能进行重新划分。train_anno_list.txt 和 val_anno_list.txt 的具体填写格式如图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/anno_list1.png' width='600px'>
<br>
<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/anno_list1.png' width='600px'>
</center>

9. 经过整理得到的最终目录结构如下：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/instance_segmentation_dataset_prepare/directory_structure.png' width='600px'>
</center>

10. 将 fruit 目录打包压缩为 .tar 或 .zip 格式压缩包即可得到水果实例分割标准 labelme 格式数据集。