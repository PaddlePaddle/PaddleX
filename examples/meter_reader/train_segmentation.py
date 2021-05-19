import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex.seg import transforms

# 下载和解压表盘分割数据集
meter_seg_dataset = 'https://bj.bcebos.com/paddlex/examples/meter_reader/datasets/meter_seg.tar.gz'
pdx.utils.download_and_decompress(meter_seg_dataset, path='./')

# 定义训练和验证时的transforms
train_transforms = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.RandomHorizontalFlip(prob=0.5),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize([512, 512]),
    transforms.Normalize(),
])
# 定义训练和验证所用的数据集
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-segdataset
train_dataset = pdx.datasets.SegDataset(
    data_dir='meter_seg/',
    file_list='meter_seg/train.txt',
    label_list='meter_seg/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.SegDataset(
    data_dir='meter_seg/',
    file_list='meter_seg/val.txt',
    label_list='meter_seg/labels.txt',
    transforms=eval_transforms)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标
# VisualDL启动方式: visualdl --logdir output/deeplab/vdl_log --port 8001
# 浏览器打开 https://0.0.0.0:8001即可
# 其中0.0.0.0为本机访问，如为远程服务, 改成相应机器IP
#
# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#paddlex-seg-deeplabv3p
model = pdx.seg.DeepLabv3p(
    num_classes=len(train_dataset.labels), backbone='Xception65')
model.train(
    num_epochs=20,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.1,
    pretrain_weights='COCO',
    save_interval_epochs=5,
    save_dir='output/meter_seg',
    use_vdl=True)
