import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddle.fluid as fluid
from paddlex.cls import transforms
import paddlex as pdx

# 下载和解压蔬菜分类数据集
veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

# 定义训练和验证时的transforms
train_transforms = transforms.Compose(
    [transforms.RandomCrop(crop_size=224),
     transforms.Normalize()])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])

# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/train_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=eval_transforms)

# PaddleX支持自定义构建优化器
step_each_epoch = train_dataset.num_samples // 32
learning_rate = fluid.layers.cosine_decay(
    learning_rate=0.025, step_each_epoch=step_each_epoch, epochs=10)
optimizer = fluid.optimizer.Momentum(
    learning_rate=learning_rate,
    momentum=0.9,
    regularization=fluid.regularizer.L2Decay(4e-5))

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/resnet50/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP
model = pdx.cls.ResNet50(num_classes=len(train_dataset.labels))
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    optimizer=optimizer,
    save_dir='output/resnet50',
    use_vdl=True)
