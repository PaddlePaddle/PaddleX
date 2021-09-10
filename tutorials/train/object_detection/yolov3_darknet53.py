import paddlex as pdx
from paddlex import transforms as T

# 下载和解压昆虫检测数据集
dataset = 'https://bj.bcebos.com/paddlex/datasets/insect_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/docs/apis/transforms/transforms.md
train_transforms = T.Compose([
    T.MixupImage(mixup_epoch=250), T.RandomDistort(),
    T.RandomExpand(im_padding_value=[123.675, 116.28, 103.53]), T.RandomCrop(),
    T.RandomHorizontalFlip(), T.BatchRandomResize(
        target_sizes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608],
        interp='RANDOM'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(
        608, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/docs/apis/datasets.md
train_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/train_list.txt',
    label_list='insect_det/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='insect_det',
    file_list='insect_det/val_list.txt',
    label_list='insect_det/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 初始化模型，并进行训练
# 可使用VisualDL查看训练指标，参考https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/docs/train/visualdl.md
num_classes = len(train_dataset.labels)
model = pdx.det.YOLOv3(num_classes=num_classes, backbone='DarkNet53')

# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0//docs/apis/models/detection.md
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0//docs/parameters.md
model.train(
    num_epochs=270,
    train_dataset=train_dataset,
    train_batch_size=8,
    eval_dataset=eval_dataset,
    learning_rate=0.001 / 8,
    warmup_steps=1000,
    warmup_start_lr=0.0,
    save_interval_epochs=5,
    lr_decay_epochs=[216, 243],
    save_dir='output/yolov3_darknet53')
