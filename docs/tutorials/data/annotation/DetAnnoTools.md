# 通用目标检测数据标注指南

本文档将介绍如何使用[Labelme](https://github.com/wkentaro/labelme)和[PaddleLabel](https://github.com/PaddleCV-SIG/PaddleLabel/tree/v1.0.0)标注工具完成目标检测相关单模型的数据标注。
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程，以下提供简洁版本说明：
## 1. 标注数据示例
该数据集是人工采集的数据集，数据种类涵盖了安全帽和人的头部两种类别，包含目标不同角度的拍摄照片。
图片示例：
<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/example1.png' width='255px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/example2.png' width='227px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/example3.png' width='118px'>
<br>
<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/example4.png' width='231px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/example5.png' width='197px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/example6.png' width='173px'>
</center>

## 2. Labelme标注工具使用
### 2.1. Labelme标注工具介绍
Labelme 是一个 python 语言编写，带有图形界面的图像标注软件。可用于图像分类，目标检测，图像分割等任务，在目标检测的标注任务中，标签存储为 JSON 文件。
### 2.2. Labelme安装
为避免环境冲突，建议在 conda 环境下安装。
```shell
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```
### 2.3. Labelme的标注过程
#### 2.3.1. 准备待标注数据
1. 创建数据集根目录，如hemlet
2. 在hemlet中创建images目录（必须为images目录），并将待标注图片存储在images目录下，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/image_dir.png' width='600px'>
</center>

3. 在hemlet文件夹中创建待标注数据集的类别标签文件label.txt，并在label.txt中按行写入待标注数据集的类别。以安全帽检测数据集的label.txt为例，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/label_txt.png' width='600px'>
</center>

#### 2.3.2. 启动Labelme
终端进入到带标注数据集根目录，并启动labelme标注工具
```shell
cd path/to/hemlet
labelme images --labels label.txt --nodata --autosave --output annotations
```
* --labels 类别标签路径。
* --nodata 停止将图像数据存储到JSON文件。
* --autosave 自动存储
* --ouput 标签文件存储路径
#### 2.3.3. 开始图片标注
1. 启动 labelme 后如图所示：


<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/labelme.png' width='600px'>
</center>

2. 点击"编辑"选择标注类型

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/edit.png' width='600px'>
</center>

3. 选择创建矩形框

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/rectangle.png' width='200px'>
</center>

4. 在图片上拖动十字框选目标区域

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/select_target_area.png' width='600px'>
</center>

5. 再次点击选择目标框类别

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/select_category.png' width='200px'>
</center>

6. 标注好后点击存储。（若在启动labelme时未指定--output字段，会在第一次存储时提示选择存储路径，若指定--autosave字段使用自动保存，则无需点击存储按钮）

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/save.png' width='100px'>
</center>

7. 然后点击"Next Image"进行下一张图片的标注

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/next_image.png' width='100px'>
</center>

8. 最终标注好的标签文件如图所示

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/annotation_result.png' width='600px'>
</center>

9. 调整目录得到安全帽检测标准labelme格式数据集
  a. 在数据集根目录创建train_anno_list.txt和val_anno_list.txt两个文本文件，并将annotations目录下的全部json文件路径按一定比例分别写入train_anno_list.txt和val_anno_list.txt，也可全部写入到train_anno_list.txt同时创建一个空的val_anno_list.txt文件，待上传零代码使用数据划分功能进行重新划分。train_anno_list.txt和val_anno_list.txt的具体填写格式如图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/anno_list.png' width='600px'>
</center>

  b. 经过整理得到的最终目录结构如下：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/directory_structure.png' width='600px'>
</center>

  c. 将hemlet目录打包压缩为.tar或.zip格式压缩包即可得到安全帽检测标准labelme格式数据集
## 3. PaddleLabel 使用
### 3.1.1. 安装与运行
为避免环境冲突，建议创建一个干净的conda环境：
```shell
conda create -n paddlelabel python=3.11
conda activate paddlelabel
```
同样可以通过pip一键安装
```shell
pip install --upgrade paddlelabel
pip install a2wsgi uvicorn==0.18.1 
pip install connexion==2.14.1
pip install Flask==2.2.2
pip install Werkzeug==2.2.2 
```
安装成功后，可以在终端使用如下指令启动 PaddleLabel
paddlelabel  # 启动paddlelabel
pdlabel # 缩写，和paddlelabel完全相同
PaddleLabel 启动后会自动在浏览器中打开网页，接下来可以根据任务开始标注流程了。详细操作说明可参考 快速使用 文档。
### 3.1.2. PaddleLabel的标注过程
1. 打开自动弹出的网页，点击样例项目，点击目标检测

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/welcome.png' width='600px'>
</center>

2. 填写项目名称，数据集路径，注意路径是本地机器上的 绝对路径。完成后点击创建。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/create_project.png' width='600px'>
</center>

3. 首先定义需要标注的类别，以版面分析为例，提供10个类别， 每个类别有唯一对应的id
|类别名|类别id|
|---|---|
|安全帽|hemlet|
|人的头部| persion|
点击添加类别，创建所需的类别名

4. 开始标注
  a. 首先选择需要标注的标签
  b. 点击左侧的矩形选择按钮
  c. 在图片中框选需要区域，注意按语义进行分区，如出现多栏情况请分别标注多个框
  d. 完成标注后，右下角会出现标注结果，可以检查标注是否正确。
  e. 全部完成之后点击 项目总览

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/overview.png' width='600px'>
</center>

5. 导出标注文件
  a. 在项目总览中按需求划分数据集，然后点击导出数据集

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/export_dataset.png' width='600px'>
</center>

  b. 填写导出路径和导出格式，导出路径依然是一个绝对路径，导出格式请选择coco

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/select_format.png' width='600px'>
</center>

  c. 导出成功后，在指定的路径下就可以获得标注文件。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/labeled_file.png' width='600px'>
</center>

6. 调整目录得到安全帽检测标准coco格式数据集
  a. 并将三个json文件以及image目录进行重命名，对应关系如下：

<center>

|源文件(目录)名|重命名后文件(目录)名|
|:--------:|:--------:|
|train.json|instance_train.json|
|val.json|instance_train.json|
|test.json|instance_test.json|
|image|images|
</center>
  b. 在数据集根目录创建annotations目录，并将json文件全部移动到annotations目录下，得到最后的数据集目录如下：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/object_detection_dataset_prepare/directory_structure2.png' width='600px'>
</center>

  c. 将hemlet目录打包压缩为.tar或.zip格式压缩包即可得到安全帽检测标准coco格式数据集

