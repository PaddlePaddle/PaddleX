# PaddleX Video Detection Task Module Data Annotation Tutorial

This document will guide you through the process of annotating data for video detection tasks.

## 1. Annotation

### 1.1 Data Preparation

Collect the raw video data and organize each video into different folders based on their categories. Supported video formats include '.mp4', '.avi', '.mov', '.mkv'. For example:

```bash
dataset_dir  # Root directory of the dataset; the directory name can be changed
├── class1    # Directory for storing videos of each category; the directory name can be changed, containing multiple video files
├── class2
├── ...

```

### 1.2 Video Data Conversion

The raw video data needs to be converted into image frames. You can use the [convert_video_to_images.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/video_det_dataset_prepare/convert_video_to_images.py) script and run the following command:

```bash
python convert_video_to_images.py dataset_dir dataset_dir/rgb-images
```
Here, `dataset_dir` is the directory with videos to be annotated from section 1.1. `rgb-images` is the directory where the converted image frames will be saved, with each video's images stored in a separate subdirectory.

### 1.3 Image Frame Data Annotation

#### Annotation Process

* For annotation, you can refer to the [Object Detection Annotation Documentation](../cv_modules/object_detection.md). Use `Labelme` or `PaddleLabel` to annotate the bounding boxes for each image, and save them in the coco.json format.

* After completing the annotation, you need to organize the annotation files into the format required in section 2. Data Format. Use the [convert_coco_to_txt.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/video_det_dataset_prepare/convert_coco_to_txt.py) script to convert the coco.json annotation files into txt format. Place these files in the `labels` directory, with each annotation file corresponding to the image frames of a single video.


```bash
python convert_coco_to_txt.py coco.json dataset_dir
```
 `dataset_dir` is the directory where you want to save the txt files.
 `coco.json` is the annotation file in coco format.

##  2. Data Format

* The directory structure for the final dataset should be as follows:

```bash
 dataset_dir    # Root directory of the dataset; the directory name can be changed
├── rgb-images     # Directory for saving video images; the directory name cannot be changed
│   ├── class1    # Directory for storing videos of each category; the directory name can be changed
│   │   ├── video1_images  # Directory for storing images from each video in the category; the directory name can be changed
│   │   │   ├── 00001.jpg  # Image naming rule: frame number.jpg, e.g., 00001.jpg
│   │   │   ├── 00002.jpg
│   │   ├── video2_images
│   ├── class2
│   ├── ...
├── labels       # Directory for saving labels of the video images; the directory name cannot be changed and corresponds to the subdirectory names in rgb-images
│   ├── class1    # Directory for storing video labels of each category; the directory name can be changed
│   │   ├── video1_images  # Directory for storing image labels from each video in the category; the directory name can be changed
│   │   │   ├── 00001.txt  # Each txt file corresponds to the label of an image, with each line formatted as: classid x1 y1 x2 y2
│   │   │   ├── 00002.txt
│   │   ├── video2_images
│   ├── class2
│   ├── ...
├── label_map.txt # File mapping label ids to category names; the file name cannot be changed. Each line provides a category id and name, e.g., 1 Basketball
├── train.txt     # Training set annotation file, with each line providing the path to a video image label, e.g., labels/Biking/v_Biking_g02_c01/00228.txt
└── val.txt       # Validation set annotation file, with each line providing the path to a video image label
```

The annotation files use an image format. Please refer to the above specifications to prepare your data. Additionally, you can refer to the [example dataset](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/video_det_examples.tar) for guidance.
