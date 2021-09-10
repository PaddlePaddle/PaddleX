# 数据划分

在模型进行训练时，我们需要划分训练集，验证集和测试集，可直接使用paddlex命令将数据集随机划分。如果数据已经划分过，该步骤可跳过。

> 注：如您使用PaddleX可视化客户端进行模型训练，数据集划分功能集成在客户端内，无需自行使用paddlex命令划分

## 图像分类

使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集:

```commandline
paddlex --split_dataset --format ImageNet --dataset_dir MyDataset --val_value 0.2 --test_value 0.1
```

划分好的数据集会额外生成`labels.txt`, `train_list.txt`, `val_list.txt`, `test_list.txt`四个文件，之后可直接进行训练。


- [图像分类任务训练示例代码](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/tutorials/train/image_classification/mobilenetv2.py)

## 目标检测

使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集:

```commandline
paddlex --split_dataset --format VOC --dataset_dir D:\MyDataset --val_value 0.2 --test_value 0.1
```
执行上面命令行，会在`D:\MyDataset`下生成`labels.txt`, `train_list.txt`, `val_list.txt`和`test_list.txt`，分别存储类别信息，训练样本列表，验证样本列表，测试样本列表


- [目标检测任务训练示例代码](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/tutorials/train/object_detection/yolov3_mobilenetv1.py)

## 实例分割

使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集:

```commandline
paddlex --split_dataset --format COCO --dataset_dir D:\MyDataset --val_value 0.2 --test_value 0.1
```
执行上面命令行，会在`D:\MyDataset`下生成`train.json`, `val.json`, `test.json`，分别存储训练样本信息，验证样本信息，测试样本信息


- [实例分割任务训练示例代码](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/tutorials/train/instance_segmentation/mask_rcnn_r50_fpn.py)

## 语义分割

使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集:
```commandline
paddlex --split_dataset --format SEG --dataset_dir D:\MyDataset --val_value 0.2 --test_value 0.1
```
执行上面命令行，会在`D:\MyDataset`下生成`train_list.txt`, `val_list.txt`, `test_list.txt`，分别存储训练样本信息，验证样本信息，测试样本信息


- [语义分割任务训练示例代码](https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/tutorials/train/semantic_segmentation/deeplabv3p_xception65.py)
