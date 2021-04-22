import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex import transforms
import cv2

dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250), transforms.RandomDistort(),
    transforms.RandomExpand(), transforms.RandomCrop(),
    transforms.RandomHorizontalFlip(), transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]), transforms.BatchRandomResize(
            target_size=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
            interp='RANDOM',
            keep_ratio=False)
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        608, 608, interp=cv2.INTER_CUBIC), transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/train_list.txt',
    label_list='insect_det/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/val_list.txt',
    label_list='insect_det/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(
    num_classes=num_classes,
    backbone='DarkNet53',
    anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119],
             [116, 90], [156, 198], [373, 326]],
    anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]])

model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_darknet53')
