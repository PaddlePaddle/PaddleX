# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'

from paddlex.det import transforms
import paddlex as pdx

# 下载和解压铝材缺陷检测数据集
aluminum_dataset = 'https://bj.bcebos.com/paddlex/examples/industrial_quality_inspection/datasets/aluminum_inspection.tar.gz'
pdx.utils.download_and_decompress(aluminum_dataset, path='./')

# API说明 https://paddlex.readthedocs.io/zh_CN/develop/apis/transforms/det_transforms.html
train_transforms = transforms.Compose([
    transforms.RandomDistort(), transforms.RandomCrop(),
    transforms.RandomHorizontalFlip(), transforms.ResizeByShort(
        short_size=[800], max_size=1333), transforms.Normalize(
            mean=[0.5], std=[0.5]), transforms.Padding(coarsest_stride=32)
])

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(
        short_size=800, max_size=1333),
    transforms.Normalize(),
    transforms.Padding(coarsest_stride=32),
])

# 定义训练和验证所用的数据集
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/datasets.html#paddlex-datasets-vocdetection
train_dataset = pdx.datasets.VOCDetection(
    data_dir='aluminum_inspection',
    file_list='aluminum_inspection/train_list.txt',
    label_list='aluminum_inspection/labels.txt',
    transforms=train_transforms,
    num_workers=8,
    shuffle=True)
eval_dataset = pdx.datasets.VOCDetection(
    data_dir='aluminum_inspection',
    file_list='aluminum_inspection/val_list.txt',
    label_list='aluminum_inspection/labels.txt',
    num_workers=8,
    transforms=eval_transforms)

# 把背景图片加入训练集中
train_dataset.add_negative_samples(
    image_dir='./aluminum_inspection/train_wu_xia_ci')

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://paddlex.readthedocs.io/zh_CN/develop/train/visualdl.html
# num_classes 需要设置为包含背景类的类别数，即: 目标类别数量 + 1
num_classes = len(train_dataset.labels) + 1

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#paddlex-det-fasterrcnn
model = pdx.det.FasterRCNN(
    num_classes=num_classes,
    backbone='ResNet50_vd_ssld',
    with_dcn=True,
    fpn_num_channels=64,
    with_fpn=True,
    test_pre_nms_top_n=500,
    test_post_nms_top_n=300)

# API说明: https://paddlex.readthedocs.io/zh_CN/develop/apis/models/detection.html#id1
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=80,
    train_dataset=train_dataset,
    train_batch_size=10,
    eval_dataset=eval_dataset,
    learning_rate=0.0125,
    lr_decay_epochs=[60, 70],
    warmup_steps=1000,
    save_dir='output/faster_rcnn_r50_vd_dcn',
    use_vdl=True)
