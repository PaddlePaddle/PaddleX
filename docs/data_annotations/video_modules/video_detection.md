---
comments: true
---

# PaddleX视频检测任务模块数据标注教程

本文档将介绍如何完成视频检测相关任务的数据标注。

## 1. 标注

### 1.1 数据准备

收集原始视频数据，讲每个视频按照类别分别放在不同的文件夹中, 支持'.mp4', '.avi', '.mov', '.mkv'格式视频。例如：

```bash
dataset_dir  # 数据集根目录，目录名称可以改变
├── class1    # 每个类别的视频图像的保存目录，目录名称可以改变,目录下有多个视频文件
├── class2
├── ...

```

### 1.2 视频数据转换

原始视频数据需要转换成图像帧，可以使用[convert_video_to_images.py]（https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/video_det_dataset_prepare/convert_video_to_images.py）脚本，执行以下命令：

```bash
python convert_video_to_images.py dataset_dir dataset_dir/rgb-images
```
 `dataset_dir` 为1.1中待标注的视频目录。
 `rgb-images` 为转换后的图像帧保存目录，每个视频的图像保存在一个同个目录下。

### 1.2 图像帧数据标注

### 标注过程

* 标注可以参考[目标检测标注文档](../cv_modules/object_detection.md)，使用`Labelme`或者 `PaddleLabel` 把每张图的待检测框标注出来, 并存为coco.json的数据格式。

* 标注完成后，需要将标注文件整理为下面的2.数据格式中要求的格式，使用 [convert_coco_to_txt.py]（https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/video_det_dataset_prepare/convert_coco_to_txt.py）脚本，将coco.json的标注文件转为txt格式, 放在`labels`目录下，每个标注文件对应一个视频的图像帧。


```bash
python convert_coco_to_txt.py coco.json dataset_dir
```
 `dataset_dir` 为1.1中待标注的视频目录。
 `coco.json` 为标注后保存的coco.json文件。


##  2. 数据格式
* PaddleX 针对视频检测任务定义的数据集，组织结构和标注格式如下：

```bash
dataset_dir    # 数据集根目录，目录名称可以改变
├── rgb-images     # 视频图像的保存目录，目录名称不可以改变
│   ├── class1    # 每个类别的视频图像的保存目录，目录名称可以改变
│   │   ├── video1_images  # 该类别下每个视频的图像保存目录，目录名称可以改变
│   │   │   ├── 00001.jpg  # 图像命名规则为：帧号.jpg，例如：00001.jpg。
│   │   │   ├── 00002.jpg
│   │   ├── video2_images
│   ├── class2
│   ├── ...
├── labels       # 视频图像的标签保存目录，目录名称不可以改变，与rgb-images的子目录名称对应
│   ├── class1    # 每个类别的视频图像的保存目录，目录名称可以改变
│   │   ├── video1_images  # 该类别下每个视频的图像标签保存目录，目录名称可以改变
│   │   │   ├── 00001.txt  # 每个txt文件对应一个图像的标签，每行内容举例：classid x1 y1 x2 y2。
│   │   │   ├── 00002.txt
│   │   ├── video2_images
│   ├── class2
│   ├── ...
├── label_map.txt # 标注id和类别名称的对应关系，文件名称不可改变。每行给出类别id和类别名称，内容举例：1 Basketball
├── train.txt     # 训练集标注文件，每行给出视频图像的标签路径，内容举例：labels/Biking/v_Biking_g02_c01/00228.txt
└── val.txt       # 验证集标注文件，每行给出视频图像的标签路径

```
标注文件采用图像格式。请大家参考上述规范准备数据，此外可以参考[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/video_det_examples.tar)。
