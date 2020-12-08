# Plot Detection ChangeDet

## Dataset folder structure

In PaddleX, the annotation files are png files**.**It is recommended that users organize the dataset in the following way: The original landscape maps of the same plot at different periods are placed in the same directory, such as JPEGImages`. The marked png files with the same name are placed in the same directory, such as `Annotations`.`
```
MyDataset/ # Semantic segmentation dataset root directory --JPEGImages/ # The directory where the original image files are located, containing images of the same object in both early stage and late stage |--1_1.jpg |--1_2.jpg |--2_1.jpg |--2_2.jpg |--. . . |--. . . | |--Annotations/ # Mark the directory where the file is located. |--1.png |--2.png |--. . . |--. . .
```
Original landscape images of the same plot at different times, such as 1_1.jpg and 1_2.jpg, which can be RGB color images, grayscale maps, or multi-channel images in tiff format. Semantically segmented annotated images, for example, 1.png, It is the single channel image. Pixel annotation categories should start from 0 in the ascending order (0 means background), for example, 0, 1, 2, 3 mean four categories. There are up to 255 categories (the pixel 255 is not involved in training and evaluation).

## Divide the training set and validation sets

**To facilitate training`, `prepare `train_list.txt , `val_list.txt` and labels.txt` files in the `MyDataset directory, indicating training set list, validation set list and category labels list, respectively.` **

**labels.txt **

labels.txt: lists all the categories. The corresponding line number of the category represents the id of the category during the training of the model (the line number starts counting from 0), for example, labels.txt has the following content:
```
unchanged changed
```
Indicates that there are two segmentation categories in the detection dataset, namely, `un`changed` and changed. In the model training, the category id corresponding to `unchanged` is 0, changed is 1, and so on. If you don’t know the specific category label, you can directly enter labels.txt one by one, 0, 1, 2…`.`.`序列即可。

**train_list.txt **

train_list.txt lists the collection of images used for training. The corresponding annotation files are as follows (example):
```
JPEGImages/1_1.jpg JPEGImages/1_2.jpg Annotations/1.png JPEGImages/2_1.jpg JPEGImages/2_2.jpg Annotations/2.png . . . . . .
```
The first and second columns correspond to the relative paths of the original image relative to `MyDataset` for different periods of the same plot, and the third column is the relative path of the labeled file relative to MyDataset``

**val_list.txt **

val_list lists the image integration used for validation. Its corresponding annotation file has the same format as val_list.txt.

## PaddleX dataset loading

[sample code (computing)](https://github.com/PaddlePaddle/PaddleX/blob/develop/examples/change_detection/train.py)
