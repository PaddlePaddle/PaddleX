# PaddleDetection Object Detection

## Dataset folder structure

In PaddleX, the object detection supports the PascalVOC dataset format. It is recommended that users organize the dataset in the following way: The original images are placed in the same directory, for example, `JPEGImages`. The marked xml files with the same name are placed in the same directory, for example, `Annotations`.
```
MyDataset/ # Object detection dataset root directory |--JPEGImages/ # The directory where the original image files are located. |--1.jpg |--2.jpg |--. . . |--. . . | |--Annotations/ # Mark the directory where the file is located. |--1.xml |--2.xml |--. . . |--. . .
```

## Divide the training set and validation sets

**To facilitate training`, `prepare `train_list.txt , `val_list.txt` and labels.txt` files in the `MyDataset directory, indicating training set list, validation set list and category labels list, respectively.`Click to download the object detection example dataset** [](https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz)

> Note: You can also use PaddleX's own tool to randomly divide the dataset. After the dataset is organized in the above format, run the following commands to quickly complete the random division of the dataset, where val_value indicates the ratio of the validation set, and test_value indicates the ratio of the test set (which can be 0), and the remaining ratio is used for the training set. ****
> ```
> paddlex --split_dataset --format VOC --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
> ```

**labels.txt **

labels.txt: lists all the categories. The corresponding line number of the category represents the id of the category during the training of the model (the line number starts counting from 0), for example, labels.txt has the following content:
```
dog cat snake
```
There are three object `cat`egories in the detection dataset`,` namely, `dog`, cat and `snake`, and the corresponding category id is 0 for dog, 1 for cat, and so on.``

**train_list.txt **

train_list.txt lists the collection of images used for training. The corresponding annotation files are as follows (example):
```
JPEGImages/1.jpg Annotations/1.xml JPEGImages/2.jpg Annotations/2.xml . . . . . .
```
The first column is the relative path of the original image relative to `MyDataset`, and the second column is the relative path of the labeled file relative to MyDataset``

**val_list.txt **

val_list lists the image integration used for validation. Its corresponding annotation file has the same format as val_list.txt.

## PaddleX dataset loading
Example codes are as follows:
```
import paddlex as pdx from paddlex.det import transforms train_transforms = transforms. Compose([ transforms. RandomHorizontalFlip(), transforms. Normalize(), transforms. ResizeByShort(short_size=800, max_size=1333), transforms. Padding(coarsest_stride=32) ]) eval_transforms = transforms. Compose([ transforms. Normalize(), transforms. ResizeByShort(short_size=800, max_size=1333), transforms. Padding(coarsest_stride=32), ]) train_dataset = pdx.datasets. VOCDetection( data_dir = '. /MyDataset', file_list='. /MyDataset/train_list.txt', label_list='. /MyDataset/labels.txt', transforms=train_transforms) eval_dataset = pdx.datasets. VOCDetection( data_dir = '. /MyDataset', file_list='. /MyDataset/val_list.txt', label_list='MyDataset/labels.txt', transforms=eval_transforms)
```
