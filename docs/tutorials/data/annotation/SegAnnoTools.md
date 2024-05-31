# 通用语义分割数据标注指南

本文档将介绍如何使用 [Labelme](https://github.com/wkentaro/labelme) 标注工具完成语义分割相关单模型的数据标注。
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程，以下提供简洁版本说明。
## 1. 标注数据示例
该数据集是人工采集的街景数据集，数据种类涵盖了车辆和道路两种类别，包含目标不同角度的拍摄照片。
图片示例：
<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/example1.png' width='200px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/example2.png' width='200px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/example3.png' width='200px'>
<br>
<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/example4.png' width='200px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/example5.png' width='200px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/example6.png' width='200px'>
</center>

## 2. Labelme标注工具使用
### 2.1. Labelme标注工具介绍
Labelme 是一个 python 语言编写，带有图形界面的图像标注软件。可用于图像分类，目标检测，语义分割等任务，在语义分割的标注任务中，标签存储为 JSON 文件。
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
1. 创建数据集根目录，如 seg_dataset
2. 在 seg_dataset 中创建 images 目录（目录名称可修改，但要保持后续命令的图片目录名称正确），并将待标注图片存储在 images 目录下，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/image_dir.png' width='600px'>
</center>

#### 2.3.2. 启动Labelme
终端进入到待标注数据集根目录，并启动 labelme 标注工具。
```
# Windows
cd C:\path\to\seg_dataset
# Mac/Linux
cd path/to/seg_dataset
```
```shell
labelme images --nodata --autosave --output annotations
```
* --nodata 停止将图像数据存储到JSON文件
* --autosave 自动存储
* --ouput 标签文件存储路径
#### 2.3.3. 开始图片标注
1. 启动 labelme 后如图所示：


<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/labelme.png' width='600px'>
</center>

2. 点击"编辑"选择标注类型

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/edit.png' width='600px'>
</center>

3. 选择创建多边形

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/polygons.png' width='200px'>
</center>

4. 在图片上绘制目标轮廓

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/select_target_area.png' width='600px'>
</center>

5. 出现如下左图所示轮廓线闭合时，弹出类别选择框，可输入或选择目标类别

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/finish_select.png' width='380px'><img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/select_category.png' width='220px'>
</center>

通常情况下，只需要标注前景目标并设置标注类别即可，其他像素默认作为背景。如需要手动标注背景区域，**类别必须设置为 \_background\_**，否则格式转换数据集会出现错误。
对于图片中的噪声部分或不参与模型训练的部分，可以使用 **\_\_ignore\_\_** 类，模型训练时会自动跳过对应部分。
针对带有空洞的目标，在标注完目标外轮廓后，再沿空洞边缘画多边形，并将空洞指定为特定类别，如果空洞是背景则指定为 \_background\_，示例如下：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/background.png' width='600px'>
</center>

6. 标注好后点击存储。（若在启动 labelme 时未指定--output 字段，会在第一次存储时提示选择存储路径，若指定--autosave 字段使用自动保存，则无需点击存储按钮）

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/save.png' width='100px'>
</center>


7. 然后点击 "Next Image" 进行下一张图片的标注

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/next_image.png' width='100px'>
</center>

8. 最终标注好的标签文件如图所示

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/annotation_result.png' width='600px'>
</center>

9. 调整目录得到安全帽检测标准labelme格式数据集

    a. 在数据集根目录 seg_datset 下载并执行[目录整理脚本](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/format_seg_labelme_dataset.py)。执行脚本后的 train_anno_list.txt 和 val_anno_list.txt 中具体内容如图所示：


    ```
    python format_seg_labelme_dataset.py
    ```

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/anno_list.png' width='600px'>
</center>

  &emsp;&emsp;
  b. 经过整理得到的最终目录结构如下：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/semantic_segmentation_dataset_prepare/directory_structure.png' width='600px'>
</center>

  &emsp;&emsp;
  c. 将 seg_dataset 目录打包压缩为 .tar 或 .zip 格式压缩包即可得到语义分割标准 labelme 格式数据集

