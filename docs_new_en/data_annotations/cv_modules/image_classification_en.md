# PaddleX Image Classification Task Module Data Annotation Tutorial

This document will introduce how to use the [Labelme](https://github.com/wkentaro/labelme) annotation tool to complete data annotation for image classification tasks with a single model. 
Click the link above to refer to the homepage documentation for installing the data annotation tool and viewing detailed usage instructions.

## 1. Labelme Annotation
### 1.1 Introduction to Labelme Annotation Tool
`Labelme` is an image annotation software written in `python` with a graphical user interface. It can be used for tasks such as image classification, object detection, and image segmentation. For instance segmentation annotation tasks, labels are stored as `JSON` files.

### 1.2 Installation of Labelme
To avoid environment conflicts, it is recommended to install in a `conda` environment.

```bash
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```

### 1.3 Labelme Annotation Process
#### 1.3.1 Prepare Data for Annotation
* Create a root directory for the dataset, e.g., `pets`.
* Create an `images` directory (must be named `images`) within `pets` and store the images to be annotated in the `images` directory, as shown below:

![alt text](/tmp/images/data_prepare/image_classification/01.png)

* Create a category label file `flags.txt` in the `pets` folder for the dataset to be annotated, and write the categories of the dataset to be annotated line by line in `flags.txt`. Taking a cat and dog classification dataset's `flags.txt` as an example, as shown below:

![alt text](/tmp/images/data_prepare/image_classification/02.png)

#### 1.3.2 Start Labelme
Navigate to the root directory of the dataset to be annotated in the terminal and start the `labelme` annotation tool.

```bash
cd path/to/pets
labelme images --nodata --autosave --output annotations --flags flags.txt
```
* `flags` creates classification labels for images, passing in the path to the labels.
* `nodata` stops storing image data in JSON files.
* `autosave` enables automatic saving.
* `output` specifies the storage path for label files.

#### 1.3.3 Begin Image Annotation
* After starting `labelme`, it will look like this:

![alt text](/tmp/images/data_prepare/image_classification/03.png)
* Select the category in the `Flags` interface.

![alt text](/tmp/images/data_prepare/image_classification/04.png)

* After annotation, click Save. (If `output` is not specified when starting `labelme`, it will prompt to select a save path upon the first save. If `autosave` is specified, there is no need to click the Save button).

![alt text](/tmp/images/data_prepare/image_classification/05.png)
* Then click `Next Image` to annotate the next image.

![alt text](/tmp/images/data_prepare/image_classification/06.png)

* After annotating all images, use the [convert_to_imagenet.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/convert_to_imagenet.py) to convert the annotated dataset to the `ImageNet-1k` dataset format, generating `train.txt`, `val.txt`, and `label.txt`.

```bash
python convert_to_imagenet.py --dataset_path /path/to/dataset
```
`dataset_path` is the path to the annotated `labelme` format classification dataset.

* The final directory structure after organization is as follows:

![alt text](/tmp/images/data_prepare/image_classification/07.png)

## 2. Data Format
* For image classification tasks, PaddleX defines a dataset named **ClsDataset**, with the following organizational structure and annotation format:

```bash
dataset_dir    # Root directory of the dataset, the directory name can be changed
├── images     # Directory for saving images, the directory name can be changed, but should correspond to the content of train.txt and val.txt
├── label.txt  # Correspondence between annotation IDs and class names, the file name cannot be changed. Each line gives the class ID and class name, for example: 45 wallflower
├── train.txt  # Annotation file for the training set, the file name cannot be changed. Each line gives the image path and image class ID, separated by a space, for example: images/image_06765.jpg 0
└── val.txt    # Annotation file for the validation set, the file name cannot be changed. Each line gives the image path and image class ID, separated by a space, for example: images/image_06767.jpg 10
```
* If you already have a dataset in the following format but without annotation files, you can use [this script](https://paddleclas.bj.bcebos.com/tools/create_cls_trainval_lists.py) to generate annotation files for your existing dataset.

```bash
dataset_dir          # Root directory of the dataset, the directory name can be changed  
├── images           # Directory for saving images, the directory name can be changed
   ├── train         # Directory for the training set, the directory name can be changed
      ├── class0     # Class name, preferably meaningful, otherwise the generated class mapping file label.txt will be meaningless
         ├── xxx.jpg # Image, nested directories are supported
         ├── xxx.jpg # Image, nested directories are supported
         ...  
      ├── class1     # Class name, preferably meaningful, otherwise the generated class mapping file label.txt will be meaningless
      ...
   ├── val           # Directory for the validation set, the directory name can be changed
```
* If you are using the PaddleX 2.x version of the image classification dataset, after splitting the training set, validation set, and test set, manually rename `train_list.txt`, `val_list.txt`, `test_list.txt` to `train.txt`, `val.txt`, `test.txt`, respectively, and modify `label.txt` according to the rules.

Original `label.txt`:

```bash
classname1
classname2
classname3
...
```
Modified `label.txt`:

```bash
0 classname1
1 classname2
2 classname3
...