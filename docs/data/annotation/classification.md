# PaddleClas Image Classification

## Data folder structure

In PaddleX, image classification supports ImageNet dataset format. The dataset directory `data_dir` contains multiple folders, and the images in each folder belong to the same category. The folder name is the category name (note that the path should not contain Chinese characters and spaces). The structure example is as follows:
```
MyDataset/ # Image classification dataset root directory |--dog/ # All pictures in the current folder belong to the dog category. |--d1.jpg |--d2.jpg |--. . . |--. . . | |--. . . | |--snake/ # All pictures in the current folder belongs to the snake category. |--s1.jpg |--s2.jpg |--. . . |--. . .
```

## Divide the training set and validation sets

**To facilitate training`, `prepare `train_list.txt , `val_list.txt` and labels.txt` files in the `MyDataset directory, indicating training set list, validation set list and category labels list, respectively.`Click to download the image classification example dataset** [](https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz)


> Note: You can also use PaddleX's own tool to randomly divide the dataset. After the dataset is organized in the above format, run the following commands to quickly complete the random division of the dataset, where val_value indicates the ratio of the validation set, and test_value indicates the ratio of the test set (which can be 0), and the remaining ratio is used for the training set. ****
> ```
> paddlex --split_dataset --format ImageNet --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
> ```


**labels.txt **

labels.txt: lists all the categories. The corresponding line number of the category represents the id of the category during the training of the model (the line number starts counting from 0), for example, labels.txt has the following content:
```
dog cat snake
```
There are three `cat`egories in the dataset`,` namely `dog`, cat and `snake`. In the model training, the category id of dog is 0, cat is 1, and so on.``

**train_list.txt **

train_list.txt lists the collection of images used for training. The corresponding category ids are as follows (example):
```
dog/d1.jpg 0 dog/d2.jpg 0 cat/c1.jpg 1 . . . . . .snake/s1.jpg 2
```
The first column is the relative path to `MyDataset`, and the second column is the category id of the corresponding category for the image.

**val_list.txt **

val_list lists the image integration used for validation. The corresponding category id is in the same format as train_list.txt.

## PaddleX dataset loading
Sample codes are as follows:
```
import paddlex as pdx from paddlex.cls import transforms train_transforms = transforms. Compose([ transforms. RandomCrop(crop_size=224), transforms. RandomHorizontalFlip(), transforms. Normalize() ]) eval_transforms = transforms. Compose([ transforms. ResizeByShort(short_size=256), transforms. CenterCrop(crop_size=224), transforms. Normalize() ]) train_dataset = pdx.datasets. ImageNet( data_dir = '. /MyDataset', file_list='. /MyDataset/train_list.txt', label_list='. /MyDataset/labels.txt', transforms=train_transforms) eval_dataset = pdx.datasets. ImageNet( data_dir = '. /MyDataset', file_list='. /MyDataset/eval_list.txt', label_list='. /MyDataset/labels.txt', transforms=eval_transforms)
```
