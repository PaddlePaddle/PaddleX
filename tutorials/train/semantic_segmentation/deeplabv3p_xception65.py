import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex.seg import transforms

# 下载和解压视盘分割数据集
optic_dataset = 'https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz'
pdx.utils.download_and_decompress(optic_dataset, path='./')

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(), 
    transforms.ResizeRangeScaling(),
    transforms.RandomPaddingCrop(crop_size=512), 
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.ResizeByLong(long_size=512), 
    transforms.Padding(target_size=512),
    transforms.Normalize()
])

# 定义训练和验证所用的数据集
train_dataset = pdx.datasets.SegDataset(
    data_dir='optic_disc_seg',
    file_list='optic_disc_seg/train_list.txt',
    label_list='optic_disc_seg/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir='optic_disc_seg',
    file_list='optic_disc_seg/val_list.txt',
    label_list='optic_disc_seg/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/deeplab/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP
num_classes = len(train_dataset.labels)
model = pdx.seg.DeepLabv3p(num_classes=num_classes, backbone='Xception65')
model.train(
    num_epochs=40,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_dir='output/deeplabv3p_mobilenetv2',
    use_vdl=True)
