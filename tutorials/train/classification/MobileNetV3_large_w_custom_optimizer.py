import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddle
import paddlex as pdx
from paddlex import transforms

veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224), transforms.Normalize()
])

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

num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_large(num_classes=num_classes)

# Create a customized optimizer with CosineAnnealingDecay and warmup steps
train_batch_size = 64
num_steps_each_epoch = len(train_dataset) // train_batch_size
num_epochs = 10
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
    learning_rate=.001, T_max=num_steps_each_epoch * num_epochs)
warmup_epoch = 5
warmup_steps = warmup_epoch * num_steps_each_epoch
scheduler = paddle.optimizer.lr.LinearWarmup(
    learning_rate=scheduler,
    warmup_steps=warmup_steps,
    start_lr=0.0,
    end_lr=.001)
custom_optimizer = paddle.optimizer.Momentum(
    learning_rate=scheduler,
    momentum=.9,
    weight_decay=paddle.regularizer.L2Decay(coeff=.00002),
    parameters=model.net.parameters())

model.train(
    num_epochs=num_epochs,
    train_dataset=train_dataset,
    train_batch_size=train_batch_size,
    eval_dataset=eval_dataset,
    save_dir='output/mobilenetv3_large')
