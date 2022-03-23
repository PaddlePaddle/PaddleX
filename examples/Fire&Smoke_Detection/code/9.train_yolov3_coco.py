# coding:utf-8
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
train_transforms = T.Compose([
    T.BatchRandomResize(
        target_sizes=[
            320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704,
            736, 768
        ],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=640, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/train_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    transforms=train_transforms,
    num_workers=0,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='/home/aistudio/dataset',
    file_list='/home/aistudio/dataset/val_list.txt',
    label_list='/home/aistudio/dataset/labels.txt',
    transforms=eval_transforms,
    num_workers=0,
    shuffle=False)

# 初始化模型，并进行训练
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-yolov3
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    pretrain_weights='COCO',
    learning_rate=0.001 / 4,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    lr_decay_epochs=[216, 243],
    save_interval_epochs=5,
    save_dir='output/yolov3_darknet53_coco')
