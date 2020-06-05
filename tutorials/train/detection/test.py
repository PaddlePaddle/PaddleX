import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
train_transforms = transforms.ComposedRCNNTransforms(
    mode='train', min_max_size=[600, 1000])
eval_transforms = transforms.ComposedRCNNTransforms(
    mode='eval', min_max_size=[600, 1000])

# 定义训练所用的数据集
train_dataset = pdx.datasets.CocoDetection(
    data_dir='jinnan2_round1_train_20190305/restricted/',
    ann_file='jinnan2_round1_train_20190305/train.json',
    transforms=train_transforms,
    shuffle=True,
    num_workers=2)
# 训练集中加入无目标背景图片
train_dataset.add_negative_samples(
    'jinnan2_round1_train_20190305/normal_train_back/')

# 定义验证所用的数据集
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='jinnan2_round1_train_20190305/restricted/',
    ann_file='jinnan2_round1_train_20190305/val.json',
    transforms=eval_transforms,
    num_workers=2)

# 初始化模型，并进行训练
model = pdx.det.FasterRCNN(num_classes=len(train_dataset.labels) + 1)
model.train(
    num_epochs=17,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    train_batch_size=8,
    learning_rate=0.01,
    lr_decay_epochs=[13, 16],
    save_dir='./output')
