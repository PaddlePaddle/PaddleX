import os
from paddlex.det import transforms
import paddlex as pdx

# 下载和解压昆虫检测数据集
insect_dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(insect_dataset, path='./')

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=800, max_size=1333), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.Normalize(),
    transforms.ResizeByShort(
        short_size=800, max_size=1333),
    transforms.Padding(coarsest_stride=32),
])
# 定义训练和验证所用的数据集
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
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/faster_rcnn_r50_fpn/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
num_classes = len(train_dataset.labels) + 1
model = pdx.det.FasterRCNN(num_classes=num_classes, backbone='ResNet50_vd')
model.train(
    num_epochs=12,
    train_dataset=train_dataset,
    train_batch_size=2,
    eval_dataset=eval_dataset,
    learning_rate=0.0025,
    lr_decay_epochs=[8, 11],
    save_dir='output/faster_rcnn_r50_fpn',
    use_vdl=True)
