import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex.seg import transforms

# 下载和解压人像分割数据集
human_seg_data = 'https://paddlex.bj.bcebos.com/humanseg/data/human_seg_data.zip'
pdx.utils.download_and_decompress(human_seg_data, path='./')

# 下载和解压人像分割预训练模型
pretrain_weights = 'https://paddleseg.bj.bcebos.com/humanseg/models/humanseg_mobile_ckpt.zip'
pdx.utils.download_and_decompress(
    pretrain_weights, path='./output/human_seg/pretrain')

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.Resize([192, 192]), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose(
    [transforms.Resize([192, 192]), transforms.Normalize()])

# 定义训练和验证所用的数据集
# API说明: https://paddlex.readthedocs.io/zh_CN/latest/apis/datasets/semantic_segmentation.html#segdataset
train_dataset = pdx.datasets.SegDataset(
    data_dir='human_seg_data',
    file_list='human_seg_data/train_list.txt',
    label_list='human_seg_data/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir='human_seg_data',
    file_list='human_seg_data/val_list.txt',
    label_list='human_seg_data/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/unet/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP

# https://paddlex.readthedocs.io/zh_CN/latest/apis/models/semantic_segmentation.html#hrnet
num_classes = len(train_dataset.labels)
model = pdx.seg.HRNet(num_classes=num_classes, width='18_small_v1')
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.001,
    pretrain_weights='./output/human_seg/pretrain/humanseg_mobile_ckpt',
    save_dir='output/human_seg',
    use_vdl=True)
