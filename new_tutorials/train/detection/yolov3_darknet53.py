import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 下载和解压昆虫检测数据集
insect_dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(insect_dataset, path='./')

# 定义训练和验证时的transforms
# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/transforms/det_transforms.html#composedyolotransforms
train_transforms = transforms.ComposedYOLOv3Transforms(
    mode='train', shape=[608, 608])
eval_transforms = transforms.ComposedYOLOv3Transforms(
    mode='eval', shape=[608, 608])

# 定义训练和验证所用的数据集
# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/datasets/detection.html#vocdetection
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
# VisualDL启动方式: visualdl --logdir output/yolov3_darknet/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP

# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/models/detection.html#yolov3
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    lr_decay_epochs=[210, 240],
    save_dir='output/yolov3_darknet53',
    use_vdl=True)
