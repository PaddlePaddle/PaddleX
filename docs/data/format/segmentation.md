# Semantic Segmentation Seg

## Dataset folder structure

In PaddleX, the annotation files are png files**.**It is recommended to organize the datasets in the following way: original images are in the same directory (for example, JPEGImages`), and the annotations files with the same png are in the same directory, for example, `Annotations`. The instance is as follows:`
```
MyDataset/ # Semantic segmentation dataset root directory |--JPEGImages/ # The directory where the original image files are located. |--1.jpg |--2.jpg |--. . . |--. . . | |--Annotations/ # Mark the directory where the file is located. |--1.png |--2.png |--. . . |--. . .
```
Semantically segmented annotated images, for example, 1.png, Png: single-channel image. The pixel label category starts from 0 in the ascending order (0 means background), for example, 0, 1, 2, 3 means four categories. The annotation category can be up to 255 categories (where the pixel value 255 are not involved in training and evaluation).

## Divide the training set and validation sets

**To facilitate training`, `prepare `train_list.txt , `val_list.txt` and labels.txt` files in the `MyDataset directory, indicating training set list, validation set list and category labels list, respectively.`Click to download the semantic segmentation sample dataset** [](https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz)

> Note: You can also use PaddleX's own tool to randomly divide the dataset. After the dataset is organized in the above format, run the following commands to quickly complete the random division of the dataset, where val_value indicates the ratio of the validation set, and test_value indicates the ratio of the test set (which can be 0), and the remaining ratio is used for the training set. ****
> ```
> paddlex --split_dataset --format Seg --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
> ```

**labels.txt **

labels.txt: lists all the categories. The corresponding line number of the category represents the id of the category during the training of the model (the line number starts counting from 0), for example, labels.txt has the following content:
```
background human car
```
Indicates that there are 3 segmentation categories in the detection dataset`,` namely, `background`, `human` and `car`. In the model training, the category id corresponding to background is 0, human corresponds to 1, and so on. If you don't know the specific category label, you can directly enter labels. txt by marking 0, 1, 2 line by line.`.`. 序列即可。

**train_list.txt **

train_list.txt lists the collection of images used for training. The corresponding annotation files are as follows (example):
```
JPEGImages/1.jpg Annotations/1.png JPEGImages/2.jpg Annotations/2.png . . . . . .
```
The first column is the relative path of the original image relative to `MyDataset`, and the second column is the relative path of the labeled file relative to MyDataset``

**val_list.txt **

val_list lists the image integration used for validation. Its corresponding annotation file has the same format as val_list.txt.

## PaddleX dataset loading

Example codes are as follows:
```
import paddlex as pdx from paddlex.seg import transforms train_transforms = transforms. Compose([ transforms. RandomHorizontalFlip(), transforms. ResizeRangeScaling(), transforms. RandomPaddingCrop(crop_size=512), transforms. Normalize() ]) eval_transforms = transforms. Compose([ transforms. ResizeByLong(long_size=512), transforms. Padding(target_size=512), transforms. Normalize() ]) train_dataset = pdx.datasets. SegDataset( data_dir = '. /MyDataset', file_list='. /MyDataset/train_list.txt', label_list='. /MyDataset/labels.txt', transforms=train_transforms) eval_dataset = pdx.datasets. SegDataset( data_dir = '. /MyDataset', file_list='. /MyDataset/val_list.txt', label_list='MyDataset/labels.txt', transforms=eval_transforms)
```
