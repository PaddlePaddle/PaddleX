import paddle
import paddlex as pdx
from paddlex import transforms as T

# 下载和解压蔬菜分类数据集
veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/transforms/transforms.md
train_transforms = T.Compose(
    [T.RandomCrop(crop_size=224), T.RandomHorizontalFlip(), T.Normalize()])

eval_transforms = T.Compose([
    T.ResizeByShort(short_size=256), T.CenterCrop(crop_size=224), T.Normalize()
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/apis/datasets.md
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

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.cls.MobileNetV3_large(num_classes=num_classes)

# 自定义优化器：使用CosineAnnealingDecay
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

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/95c53dec89ab0f3769330fa445c6d9213986ca5f/paddlex/cv/models/classifier.py#L153
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=num_epochs,
    train_dataset=train_dataset,
    train_batch_size=train_batch_size,
    eval_dataset=eval_dataset,
    optimizer=custom_optimizer,
    save_dir='output/mobilenetv3_large',
    use_vdl=True)
