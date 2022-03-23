from paddlex.det import transforms
import paddlex as pdx

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/1.3.11/paddlex/cv/transforms/operators.py
train_transforms = transforms.Compose([
    transforms.MixupImage(mixup_epoch=-1),
    transforms.RandomDistort(),
    transforms.RandomExpand(),
    transforms.RandomCrop(),
    transforms.Resize(
        target_size=480, interp='RANDOM'),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(),
])

eval_transforms = transforms.Compose([
    transforms.Resize(
        target_size=480, interp='CUBIC'),
    transforms.Normalize(),
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/1.3.11/paddlex/cv/datasets/voc.py
train_dataset = pdx.datasets.VOCDetection(
    data_dir='work/dataset_reinforcing_steel_bar_counting',
    file_list='work/dataset_reinforcing_steel_bar_counting/train_list.txt',
    label_list='work/dataset_reinforcing_steel_bar_counting/label_list.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='work/dataset_reinforcing_steel_bar_counting',
    file_list='work/dataset_reinforcing_steel_bar_counting/val_list.txt',
    label_list='work/dataset_reinforcing_steel_bar_counting/label_list.txt',
    transforms=eval_transforms,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/tree/release/1.3.11/tutorials/train#visualdl可视化训练指标
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(
    num_classes=num_classes,
    backbone='MobileNetV1',
    label_smooth=True,
    ignore_threshold=0.7)

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/1.3.11/paddlex/cv/models/detector.py
# 各参数介绍与调整说明：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html
model.train(
    num_epochs=270,  # 训练轮次
    train_dataset=train_dataset,  # 训练数据
    eval_dataset=eval_dataset,  # 验证数据
    train_batch_size=2,  # 批大小
    pretrain_weights='COCO',  # 预训练权重
    learning_rate=0.000125,  # 学习率
    warmup_steps=1000,  # 预热步数
    warmup_start_lr=0.0,  # 预热起始学习率
    save_interval_epochs=5,  # 每5个轮次保存一次，有验证数据时，自动评估
    lr_decay_epochs=[210, 240],  # step学习率衰减
    save_dir='output/yolov3_mobilnetv1',  # 保存路径
    use_vdl=True)  # 其用visuadl进行可视化训练记录
