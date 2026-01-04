---
comments: true
---

# PaddleX Video Classification Task Module Data Annotation Tutorial

This document will introduce how to use the [BILS](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/zh-CN/annotation_tools.md) annotation tool to complete data annotation for video classification-related single models.
Click the above link to install the data annotation tool and view the detailed usage process by referring to the homepage documentation.

## 1. BILS Annotation
### 1.1 Introduction to BILS Annotation Tool
`BILS` (Baidu Intelligent Labeling System) is a video annotation software that supports tagging on the timeline and can be used for annotation tasks such as video event localization and short video classification. The user interface is simple, and the operation is easy and intuitive.
### 1.2 BILS Installation
Click the download link to download the installation package locally, and then follow the prompts to install it step-by-step.

macOS: [dmg package download](https://videotag.bj.bcebos.com/Annotation-tools/4.11-EIVideo-0.0.0.dmg)

Windows: [exe file download](https://videotag.bj.bcebos.com/Annotation-tools/EIVideo-Setup-0.0.0.exe)

Instructional video: [video download](https://videotag.bj.bcebos.com/Annotation-tools/4.11-%E4%BA%A7%E5%93%81%E8%AF%B4%E6%98%8E.mp4)
### 1.3 BILS Annotation Process
#### 1.3.1 Prepare Data to Be Annotated
* Create a root directory for the dataset, such as `video_cls`.
* Create a `videos` directory within `video_cls` and store the videos to be annotated in the `videos` directory, as shown below:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/01.png">

#### 1.3.2 BILS
Click the BILS software icon to launch the `BILS` annotation tool.

#### 1.3.3 Start Video Annotation
* After launching `BILS`, it will look like this:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/02.png">

* Click the `Settings` button in the right navigation bar, fill in the project name, and set both the project directory and dataset directory to the storage directory of the videos to be annotated, i.e., `video_cls/videos`. Click the `Update Files` button to read the videos to be annotated. After updating the files, the first video in the video folder will play automatically, and you can click the arrow icons to pause, play, or replay.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/03.png">

* Click the `Annotation` button in the right navigation bar. The default labels are `Goal, Three-pointer, Two-pointer`. If you need to create a new label, click the `Edit` icon, create a new label, and rename it. Here, `Drumming` is used as an example.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/04.png">

* Hold down the `Option` key and click somewhere in the video clip to set the start and end times of the video. Click the trash can icon to delete previously annotated labels. After determining the start and end times of the video clip, check `Drumming`, and the action category of the current time segment will be marked as `Drumming`. Finally, click the `OK` icon to save the current label to the `BILS` format annotation file.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/05.png">

* After completing the annotation of the current video, click the `Folder Mode` icon to switch to the video file mode. Then, click on the next video that needs annotation and repeat the above annotation process.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/06.png">

* After annotating all videos, click the `Export` icon to export the annotated label files.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/video_classification/07.png">

* After obtaining the exported JSON file (the default name is `ai.json`), use the [convert_to_videocls.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/video_classification_dataset_prepare/convert_to_videocls.py) script to convert the exported dataset into the `Video Classification` dataset format. Generate `train.txt`, `val.txt`, and `label.txt`.

```bash
python convert_to_videocls.py --dataset_path /path/to/dataset
```
`dataset_path` is the annotated `BILS` format classification dataset.

## 2. Data Format
* The dataset defined by PaddleX for the video classification task is named **VideoClsDataset**, with the following organization structure and annotation format:

```bash
dataset_dir    # Root directory of the dataset, the directory name can be changed
├── videos     # Directory for storing videos, the directory name can be changed, but note the correspondence with the content of train.txt and val.txt
├── label.txt  # Correspondence between annotation IDs and category names, the file name cannot be changed. Each line gives the category ID and category name, for example: 0 abseiling
├── train.txt  # Training set annotation file, the file name cannot be changed. Each line gives the video path and video category ID, separated by a space, for example: videos/Qbo_tnzfjOY.mp4 2
└── val.txt    # Validation set annotation file, the file name cannot be changed. Each line gives the video path and video category ID, separated by a space, for example: videos/3caPS4FHFF8.mp4 0
```

Annotation files are in video format. Please prepare your data by referring to the above specifications, and you can also refer to the [example dataset](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/k400_examples.tar).
