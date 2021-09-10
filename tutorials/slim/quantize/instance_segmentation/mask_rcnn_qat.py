import paddlex as pdx
from paddlex import transforms as T

# 下载和解压小度熊分拣数据集
dataset = 'https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_ins_det.tar.gz'
pdx.utils.download_and_decompress(dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0-rc/paddlex/cv/transforms/operators.py
train_transforms = T.Compose([
    T.RandomResizeByShort(
        short_sizes=[640, 672, 704, 736, 768, 800],
        max_size=1333,
        interp='CUBIC'), T.RandomHorizontalFlip(), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.ResizeByShort(
        short_size=800, max_size=1333, interp='CUBIC'), T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/release/2.0.0/paddlex/cv/datasets/coco.py#L26
train_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/train.json',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.CocoDetection(
    data_dir='xiaoduxiong_ins_det/JPEGImages',
    ann_file='xiaoduxiong_ins_det/val.json',
    transforms=eval_transforms)

# 加载模型
model = pdx.load_model('output/mask_rcnn_r50_fpn/best_model')

# 在线量化
model.quant_aware_train(
    num_epochs=6,
    train_dataset=train_dataset,
    train_batch_size=1,
    eval_dataset=eval_dataset,
    learning_rate=0.000125,
    save_dir='output/mask_rcnn_r50_fpn/quant',
    use_vdl=True)
