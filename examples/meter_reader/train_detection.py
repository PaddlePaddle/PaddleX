import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 下载和解压表计检测数据集
meter_det_dataset = 'https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_det.tar.gz'
pdx.utils.download_and_decompress(meter_det_dataset, path='./')

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=250),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(
        target_size=608, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=608, interp='CUBIC'),
    transforms.Normalize(),
])

# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.CocoDetection(
    data_dir='meter_det/train/',
    ann_file='meter_det/annotations/instance_train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='meter_det/test/',
    ann_file='meter_det/annotations/instance_test.json',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/yolov3_darknet/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP

# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/models/detection.html#yolov3
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(
    num_classes=num_classes, backbone='DarkNet53', label_smooth=True)
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.001,
    warmup_steps=4000,
    lr_decay_epochs=[210, 240],
    save_dir='output/meter_det',
    use_vdl=True)
