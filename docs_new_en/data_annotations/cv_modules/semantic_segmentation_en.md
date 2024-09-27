# PaddleX Semantic Segmentation Task Module Data Annotation Tutorial

This document will introduce how to use the [Labelme](https://github.com/wkentaro/labelme) annotation tool to complete data annotation for a single model related to semantic segmentation. Click on the link above to install the data annotation tool and view detailed usage instructions by referring to the homepage documentation.

## 1. Annotation Data Examples
This dataset is a manually collected street scene dataset, covering two categories of vehicles and roads, including photos taken from different angles of the targets. Image examples:

<div style="display: flex;">
  <img src="/tmp/images/data_prepare/semantic_seg/01.png" alt="Example Image 1">
  <img src="/tmp/images/data_prepare/semantic_seg/02.png" alt="Example Image 2">
  <img src="/tmp/images/data_prepare/semantic_seg/03.png" alt="Example Image 3">
</div>

<div style="display: flex;">
  <img src="/tmp/images/data_prepare/semantic_seg/04.png" alt="Example Image 4">
  <img src="/tmp/images/data_prepare/semantic_seg/05.png" alt="Example Image 5">
  <img src="/tmp/images/data_prepare/semantic_seg/06.png" alt="Example Image 6">
</div>

## 2. Labelme Annotation
### 2.1 Introduction to Labelme Annotation Tool
`Labelme` is an image annotation software written in `python` with a graphical interface. It can be used for tasks such as image classification, object detection, and semantic segmentation. In semantic segmentation annotation tasks, labels are stored as `JSON` files.

### 2.2 Labelme Installation
To avoid environment conflicts, it is recommended to install in a `conda` environment.

```bash
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```

### 2.3 Labelme Annotation Process
#### 2.3.1 Prepare Data for Annotation
* Create a root directory for the dataset, such as `seg_dataset`
* Create an `images` directory within `seg_dataset` (the directory name can be modified, but ensure the subsequent command's image directory name is correct), and store the images to be annotated in the `images` directory, as shown below:

![alt text](/tmp/images/data_prepare/semantic_seg/07.png)

#### 2.3.2 Launch Labelme
Navigate to the root directory of the dataset to be annotated in the terminal and launch the `labelme` annotation tool.

```bash
# Windows
cd C:\path\to\seg_dataset
# Mac/Linux
cd path/to/seg_dataset
```
```bash
labelme images --nodata --autosave --output annotations
```
* `nodata` stops storing image data in the JSON file
* `autosave` enables automatic saving
* `output` specifies the path for storing label files

#### 2.3.3 Start Image Annotation
* After launching `labelme`, it will look like this:

![alt text](/tmp/images/data_prepare/semantic_seg/08.png)
* Click "Edit" to select the annotation type

![alt text](/tmp/images/data_prepare/semantic_seg/09.png)
* Choose to create polygons
  
![alt text](/tmp/images/data_prepare/semantic_seg/10.png)
* Draw the target contour on the image

![alt text](/tmp/images/data_prepare/semantic_seg/11.png)

* When the contour line is closed as shown in the left image below, a category selection box will pop up, allowing you to input or select the target category

![alt text](/tmp/images/data_prepare/semantic_seg/12.png)
![alt text](/tmp/images/data_prepare/semantic_seg/13.png)

Typically, only the foreground objects need to be labeled with their respective categories, while other pixels are automatically considered as background. If manual background labeling is required, the **category must be set to _background_**, otherwise errors may occur during dataset format conversion. For noisy parts or irrelevant sections in the image that should not participate in model training, the **__ignore__** class can be used, and the model will automatically skip those parts during training. For objects with holes, after outlining the main object, draw polygons along the edges of the holes and assign a specific category to the holes. If the hole represents background, assign it as **_background_**. An example is shown below:

![alt text](/tmp/images/data_prepare/semantic_seg/14.png)

* After labeling, click "Save". (If the `output` field is not specified when starting `labelme`, it will prompt to select a save path upon the first save. If `autosave` is enabled, the save button is not required.)

![alt text](/tmp/images/data_prepare/semantic_seg/15.png)
* Then click "Next Image" to proceed to the next image for labeling.

![alt text](/tmp/images/data_prepare/semantic_seg/16.png)

* The final labeled file will look like this:

![alt text](/tmp/images/data_prepare/semantic_seg/17.png)

* Adjust the directory structure to obtain a standard LabelMe format dataset for safety helmet detection:
    a. Download and execute the [directory organization script](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/format_seg_labelme_dataset.py) in the root directory of your dataset, `seg_dataset`. After executing the script, the `train_anno_list.txt` and `val_anno_list.txt` files will contain content as shown:

    ```bash
    python format_seg_labelme_dataset.py
    ```
    ![alt text](/tmp/images/data_prepare/semantic_seg/18.png)
    b. The final directory structure after organization will look like this:

    ![alt text](/tmp/images/data_prepare/semantic_seg/19.png)

#### 2.3.4 Format Conversion
After labeling with `LabelMe`, the data format needs to be converted to the `Seg` data format. Below is a code example for converting data labeled using `LabelMe` according to the above tutorial.

```bash
cd /path/to/paddlex
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_dataset_to_convert.tar -P ./dataset
tar -xf ./dataset/seg_dataset_to_convert.tar -C ./dataset/

python main.py -c paddlex/configs/semantic_segmentation/PP-LiteSeg-T.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/seg_dataset_to_convert \
    -o CheckDataset.convert.enable=True \
    -o CheckDataset.convert.src_dataset_type=LabelMe
```

## Data Format
The dataset defined by PaddleX for image segmentation tasks is named **SegDataset**, with the following organizational structure and annotation format:

```ruby
dataset_dir         # Root directory of the dataset, the directory name can be changed
├── annotations     # Directory for storing annotated images, the directory name can be changed, matching the content of the manifest files
├── images          # Directory for storing original images, the directory name can be changed, matching the content of the manifest files
├── train.txt       # Annotation file for the training set, the file name cannot be changed. Each line contains the path to the original image and the annotated image, separated by a space. Example: images/P0005.jpg annotations/P0005.png
└── val.txt         # Annotation file for the validation set, the file name cannot be changed. Each line contains the path to the original image and the annotated image, separated by a space. Example: images/N0139.jpg annotations/N0139.png
```
Label images should be single-channel grayscale or single-channel pseudo-color images, and it is recommended to save them in `PNG` format. Each pixel value in the label image represents a category, and the categories should start from 0 and increase sequentially, for example, 0, 1, 2, 3 represent 4 categories. The pixel storage of the label image is 8bit, so a maximum of 256 categories are supported for labeling.

Please prepare your data according to the above specifications. Additionally, you can refer to: [Example Dataset](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/seg_optic_examples.tar)