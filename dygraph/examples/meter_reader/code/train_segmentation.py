import paddlex as pdx
from paddlex import transforms as T

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    T.Resize(target_size=512),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(target_size=512),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 下载和解压指针刻度分割数据集，如果已经预先下载，可注视掉下面两行
meter_seg_dataset = 'https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_seg.tar.gz'
pdx.utils.download_and_decompress(meter_seg_dataset, path='./')

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/datasets/seg_dataset.py#L22
train_dataset = pdx.datasets.SegDataset(
    data_dir='meter_seg',
    file_list='meter_seg/train.txt',
    label_list='meter_seg/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.SegDataset(
    data_dir='meter_seg',
    file_list='meter_seg/val.txt',
    label_list='meter_seg/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train#visualdl可视化训练指标
num_classes = len(train_dataset.labels)
model = pdx.models.DeepLabV3P(
    num_classes=num_classes, backbone='ResNet50_vd', use_mixed_loss=True)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/models/segmenter.py#L150
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=20,
    train_dataset=train_dataset,
    train_batch_size=4,
    pretrain_weights='IMAGENET',
    eval_dataset=eval_dataset,
    learning_rate=0.1,
    save_dir='output/deeplabv3p_r50vd')
