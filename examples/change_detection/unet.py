# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import paddlex as pdx
from paddlex.seg import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/seg_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomPaddingCrop(
        crop_size=256,
        im_padding_value=[127.5] * 6), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(), transforms.Normalize(
        mean=[0.5] * 6, std=[0.5] * 6, min_val=[0] * 6, max_val=[255] * 6)
])

eval_transforms = transforms.Compose([
    transforms.Padding(
        target_size=256, im_padding_value=[127.5] * 6), transforms.Normalize(
            mean=[0.5] * 6, std=[0.5] * 6, min_val=[0] * 6, max_val=[255] * 6)
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-segdataset
train_dataset = pdx.datasets.ChangeDetDataset(
    data_dir='dataset',
    file_list='dataset/train_list.txt',
    label_list='dataset/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ChangeDetDataset(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p
model = pdx.seg.UNet(num_classes=num_classes, input_channel=6)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=400,
    train_dataset=train_dataset,
    train_batch_size=16,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_interval_epochs=10,
    pretrain_weights='CITYSCAPES',
    save_dir='output/unet',
    use_vdl=True)
