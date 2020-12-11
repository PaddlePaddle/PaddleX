# Image Classification

Image classification and annotation is the most basic and simplest annotation task. Users only need to put the images belonging to the same category in the same folder, such as the directory structure shown below,
```
MyDataset/ # Image classification dataset root
|--dog/ # All pictures in the current folder belong to dog category
|  |--d1.jpg
|  |--d2.jpg
|  |--...
|  |--...
|
|--...
|
|--snake/ # All pictures in the current folder belong to snake category
|  |--s1.jpg
|  |--s2.jpg
|  |--...
|  |--...
```

## Data partition

When training the model, we need to divide the training set, verification set and test set, so we need to divide the above data. We can divide the data set into 70% training set, 20% verification set and 10% test set by using paddlex command
```
paddlex --split_dataset --format ImageNet --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
```

`labels.txt`, `train_list.txt`, `val_list.txt`, `test_list.txt` are generated from the divided data set, and then the training can be carried out directly.

> Note: if you use PaddleX visual client for model training, the data set partition function is integrated in the client, and there is no need to use command partition by yourself


- [Image Classification Task Training Example Code](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py)

