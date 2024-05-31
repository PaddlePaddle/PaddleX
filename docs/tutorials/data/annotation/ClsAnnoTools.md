# 通用图像分类数据标注指南

本文档将介绍如何使用 [Labelme](https://github.com/wkentaro/labelme) 标注工具完成图像分类相关单模型的数据标注。
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程，以下提供简洁版本说明：
## 1 Labelme 标注工具安装
### 1.1 Labelme 标注工具介绍
Labelme 是一个 python 语言编写，带有图形界面的图像标注软件。可用于图像分类，目标检测，图像分割等任务，在图像分类的标注任务中，标签存储为 JSON 文件。
### 1.2 Labelme 安装
为避免环境冲突，建议在 conda 环境下安装。
```shell
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```
## 2 Labelme 标注过程
### 2.1 准备待标注数据
1. 创建数据集根目录，如 pets。
2. 在 pets 中创建 images 目录（必须为images目录），并将待标注图片存储在 images 目录下，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/image_dir.png' width='600px'>
</center>

3. 在 pets 文件夹中创建待标注数据集的类别标签文件 flags.txt，并在 flags.txt 中按行写入待标注数据集的类别。以猫狗分类数据集的 flags.txt 为例，如下图所示：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/label_txt.png' width='600px'>
</center>

### 2.2 启动 Labelme
终端进入到待标注数据集根目录，并启动 labelme 标注工具。
```shell
cd path/to/pets
labelme images --nodata --autosave --output annotations --flags flags.txt
```
* --flags 为图像创建分类标签，传入标签路径。
* --nodata 停止将图像数据存储到 JSON 文件。
* --autosave 自动存储。
* --ouput 标签文件存储路径。
### 2.3 开始图片标注
1. 启动 labelme 后如图所示：


<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/labelme.png' width='600px'>
</center>

2. 在 Flags 界面选择类别。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/flags.png' width='300px'>
</center>

3. 标注好后点击存储。（若在启动 labelme 时未指定 --output 字段，会在第一次存储时提示选择存储路径，若指定 --autosave 字段使用自动保存，则无需点击存储按钮）。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/save.png' width='100px'>
</center>

4. 然后点击 "Next Image" 进行下一张图片的标注。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/next_image.png' width='100px'>
</center>

5. 最终标注好的标签文件如图所示。

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/annotation_result.png' width='600px'>
</center>

6. 使用 [convert_to_imagenet.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/convert_to_imagenet.py) 脚本将标注好的数据集转换为 ImageNet-1k 数据集格式，生成 train.txt，val.txt 和label.txt。

```shell
python convert_to_imagenet.py --dataset_path /path/to/dataset
```
* --dataset_path 标注的 labelme 格式分类数据集。

7. 经过整理得到的最终目录结构如下：

<center>

<img src='https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/directory_structure.png' width='600px'>
</center>

8. 将 pets 目录打包压缩为 .tar 或 .zip 格式压缩包即可得到猫狗图像分类标准 labelme 格式数据集，然后上传至 [通用图像分类产线](https://aistudio.baidu.com/pipeline/mine) 经过数据化分后即可进行训练。