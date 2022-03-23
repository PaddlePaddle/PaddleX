import paddlex as pdx
from paddlex import transforms as T

# 定义预处理变换
train_transforms = T.Compose([
    T.Resize(
        target_size=[128, 800], interp='LINEAR', keep_ratio=False),
    T.RandomHorizontalFlip(),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

eval_transforms = T.Compose([
    T.Resize(
        target_size=[128, 800], interp='LINEAR', keep_ratio=False),
    T.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 定义数据集
train_dataset = pdx.datasets.SegDataset(
    data_dir='steel',
    file_list='steel/train_list.txt',
    label_list='steel/labels.txt',
    transforms=train_transforms,
    num_workers='auto',
    shuffle=True)

eval_dataset = pdx.datasets.SegDataset(
    data_dir='steel',
    file_list='steel/val_list.txt',
    label_list='steel/labels.txt',
    transforms=eval_transforms,
    shuffle=False)

# 定义模型
num_classes = len(train_dataset.labels)
model = pdx.seg.HRNet(num_classes=num_classes, width=48)

# 训练
model.train(
    num_epochs=100,
    train_dataset=train_dataset,
    train_batch_size=48,  # 需根据实际显存大小调节
    eval_dataset=eval_dataset,
    learning_rate=0.04,
    use_vdl=True,
    save_interval_epochs=1,
    save_dir='output/hrnet')
