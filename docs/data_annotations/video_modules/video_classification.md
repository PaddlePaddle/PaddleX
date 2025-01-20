---
comments: true
---

# PaddleX视频分类任务模块数据标注教程

本文档将介绍如何使用 [BILS](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/annotation_tools.md) 标注工具完成视频分类相关单模型的数据标注。
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程。

## 1. BILS 标注
### 1.1 BILS 标注工具介绍
`BILS` (Baidu Intelligent Labeling System) 是一款支持时间轴打标签的视频标注软件，可被用于视频事件定位 、短视频分类等任务的标注工作。用户界面简约，操作简单、易上手。
### 1.2 BILS 安装
点击下载链接，将安装包下载到本地，之后按照提示一步步安装即可。

mac端: [dmg包下载](https://videotag.bj.bcebos.com/Annotation-tools/4.11-EIVideo-0.0.0.dmg)

windows端： [exe文件下载](https://videotag.bj.bcebos.com/Annotation-tools/EIVideo-Setup-0.0.0.exe)

使用教学视频: [视频下载](https://videotag.bj.bcebos.com/Annotation-tools/4.11-%E4%BA%A7%E5%93%81%E8%AF%B4%E6%98%8E.mp4)
### 1.3 BILS 标注过程
#### 1.3.1 准备待标注数据
* 创建数据集根目录，如 `video_cls`。
* 在 `video_cls` 中创建 `videos` 目录，并将待标注视频存储在 `videos` 目录下，如下图所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/01.png">

#### 1.3.2 BILS
点击BILS软件图标，启动` BILS` 标注工具

#### 1.3.3 开始图片标注
* 启动 `` BILS`` 后如图所示：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/02.png">

* 点击右侧导航栏中 `设置` 按钮，填写项目名称，并将项目目录、数据集目录均设置成标注视频的存储目录，即  `video_cls/video` 。点击更新文件按钮，读取待标注视频。更新文件后，视频文件夹中的第一个视频会自动播放，可以点击箭头图标进行暂停、播放或重播。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/03.png">

* 点击右侧导航栏中 `标注` 按钮，默认标签为 `进球、三分球、二分球`。如果需要新建标签，可以点击 `编辑` 图标，新建标签，并对其进行重命令。这里，以 `打鼓`为例。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/04.png">

*  按住 ` option` 键点击视频片段的某处，可以设置视频的起始时间和终止时间。点击垃圾桶图标可以删除之前标注的标签。确定视频片段的起始、终止时间后，勾选 `打鼓`， 即可将当前时间片段的动作类别标记为 `打鼓`。最后，点击 `确定` 图标，将当前标签保存到 `BILS` 格式的标注文件中。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/05.png">


* 当前视频标注完成后, 点击 `文件夹模式` 图标，切换到视频文件模式。然后，点击下一个需要标注的视频重复上述标注。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/06.png">


* 所有视频标注完成后，点击`导出`图标，可以导出标注好的标签文件。


<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/07.png">


* 获得导出的json文件后（默认名称是 `ai.json` ），使用 [convert_to_videocls.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/video_classification_dataset_prepare/convert_to_videocls.py) 脚本，将导出的数据集转化为 `视频分类` 数据集格式。生成 `train.txt`，`val.txt` 和`label.txt`。

```bash
python convert_to_videocls.py --dataset_path /path/to/dataset
```
 `dataset_path` 为标注的 `BILS` 格式分类数据集。


##  2. 数据格式
* PaddleX 针对视频分类任务定义的数据集，名称是 <b>VideoClsDataset</b>，组织结构和标注格式如下：

```bash
dataset_dir    # 数据集根目录，目录名称可以改变
├── videos     # 视频的保存目录，目录名称可以改变，但要注意与train.txt、val.txt的内容对应
├── label.txt  # 标注id和类别名称的对应关系，文件名称不可改变。每行给出类别id和类别名称，内容举例：0 abseiling
├── train.txt  # 训练集标注文件，文件名称不可改变。每行给出视频路径和视频类别id，使用空格分隔，内容举例：videos/Qbo_tnzfjOY.mp4 2
└── val.txt    # 验证集标注文件，文件名称不可改变。每行给出视频路径和视频类别id，使用空格分隔，内容举例：videos/3caPS4FHFF8.mp4 0
```
标注文件采用视频格式。请大家参考上述规范准备数据，此外可以参考[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/k400_examples.tar)。
