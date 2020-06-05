import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from paddlex.det import transforms
import paddlex as pdx

# 下载和解压小度熊分拣数据集
xiaoduxiong_dataset = 'https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_ins_det.tar.gz'
pdx.utils.download_and_decompress(xiaoduxiong_dataset, path='./')

# 定义训练和验证时的transforms
# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/transforms/det_transforms.html#composedrcnntransforms
train_transforms = transforms.ComposedRCNNTransforms(mode='train', min_max_size=[800, 1333])
eval_transforms = transforms.ComposedRCNNTransforms(mode='eval', min_max_size=[800, 1333])

# 定义训练和验证所用的数据集
# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/datasets/detection.html#cocodetection
train_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/val.json',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/mask_rcnn_r50_fpn/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1

# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/models/instance_segmentation.html#maskrcnn
num_classes = len(train_dataset.labels) + 1
model = pdx.det.MaskRCNN(num_classes=num_classes)
model.train(
    num_epochs=12,
    train_dataset=train_dataset,
    train_batch_size=1,
    eval_dataset=eval_dataset,
    learning_rate=0.00125,
    warmup_steps=10,
    lr_decay_epochs=[8, 11],
    save_dir='output/mask_rcnn_r50_fpn',
    use_vdl=True)
