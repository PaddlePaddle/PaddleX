import numpy as np
import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=-1), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=480, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/datasets/voc.py
train_dataset = pdx.datasets.VOCDetection(
    data_dir='work',
    file_list='work/train_list.txt',
    label_list='work/label_list.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='work',
    file_list='work/val_list.txt',
    label_list='work/label_list.txt',
    transforms=eval_transforms,
    shuffle=False)

# YOLO检测模型的预置anchor生成
# API说明: https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/tools/anchor_clustering/yolo_cluster.py
anchors = train_dataset.cluster_yolo_anchor(num_anchors=9, image_size=480)
anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/2.0.0/tutorials/train#visualdl可视化训练指标
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53', 
                       anchors= anchors.tolist() if isinstance(anchors, np.ndarray) else anchors,
                       anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]], label_smooth=True, ignore_threshold=0.6)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/models/detector.py
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=200,                     # 训练轮次
    train_dataset=train_dataset,        # 训练数据
    eval_dataset=eval_dataset,          # 验证数据
    train_batch_size=16,                # 批大小
    pretrain_weights='COCO',            # 预训练权重
    learning_rate=0.005 / 12,           # 学习率
    warmup_steps=500,                   # 预热步数
    warmup_start_lr=0.0,                # 预热起始学习率
    save_interval_epochs=5,             # 每5个轮次保存一次，有验证数据时，自动评估
    lr_decay_epochs=[85, 135],          # step学习率衰减
    save_dir='output/yolov3_darknet53', # 保存路径
    use_vdl=True)                       # 其用visuadl进行可视化训练记录