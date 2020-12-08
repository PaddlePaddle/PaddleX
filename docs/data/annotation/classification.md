# Image Classification

图像分类标注是一项最基础，最简单的标注任务，用户只需将属于同一类的图片放在同一个文件夹下即可，例如下所示目录结构，
```
MyDataset/ # 图像分类数据集根目录
|--dog/ # 当前文件夹所有图片属于dog类别
|  |--d1.jpg
|  |--d2.jpg
|  |--...
|  |--...
|
|--...
|
|--snake/ # 当前文件夹所有图片属于snake类别
|  |--s1.jpg
|  |--s2.jpg
|  |--...
|  |--...
```

## Data partition

在模型进行训练时，我们需要划分训练集，验证集和测试集，因此需要对如上数据进行划分，直接使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集
```
paddlex --split_dataset --format ImageNet --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
```

划分好的数据集会额外生成`labels.txt`, `train_list.txt`, `val_list.txt`, `test_list.txt`四个文件，之后可直接进行训练。

> 注：如您使用PaddleX可视化客户端进行模型训练，数据集划分功能集成在客户端内，无需自行使用命令划分


- [图像分类任务训练示例代码](https://github.com/PaddlePaddle/PaddleX/blob/develop/tutorials/train/image_classification/mobilenetv2.py)

