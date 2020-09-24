# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import paddlex as pdx
from paddlex.seg import transforms

# 定义训练和验证时的transforms
# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/seg_transforms.html
train_transforms = transforms.Compose([
    #transforms.ResizeStepScaling(
    #    min_scale_factor=0.5,
    #    max_scale_factor=2.,
    #    scale_step_size=0.25),
    transforms.RandomPaddingCrop(
        crop_size=769, im_padding_value=[127.5] * 6),
    #transforms.ResizeByLong(long_size=512),
    #transforms.Padding(
    #    target_size=512, im_padding_value=[127.5] * 6),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(
        mean=[0.5] * 6, std=[0.5] * 6, min_val=[0] * 6, max_val=[255] * 6)
])

eval_transforms = transforms.Compose([
    transforms.Padding(
        target_size=1000, im_padding_value=[127.5] * 6), transforms.Normalize(
            mean=[0.5] * 6, std=[0.5] * 6, min_val=[0] * 6, max_val=[255] * 6)
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-segdataset
train_dataset = pdx.datasets.ChangeDetDataset(
    data_dir='dataset',
    file_list='dataset/train_list.txt',
    label_list='dataset/labels.txt',
    transforms=train_transforms,
    num_workers=4,
    shuffle=True)
eval_dataset = pdx.datasets.ChangeDetDataset(
    data_dir='dataset',
    file_list='dataset/val_list.txt',
    label_list='dataset/labels.txt',
    num_workers=4,
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
num_classes = len(train_dataset.labels)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p
model = pdx.seg.UNet(
    num_classes=num_classes,
    input_channel=6,
    use_bce_loss=True,
    use_dice_loss=True)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#train
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=400,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_interval_epochs=10,
    pretrain_weights='CITYSCAPES',
    save_dir='output/unet_3',
    use_vdl=True)
