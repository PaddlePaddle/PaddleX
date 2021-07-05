import paddlex as pdx
from paddlex import transforms as T

# 下载和解压视盘分割数据集
optic_dataset = 'https://bj.bcebos.com/paddlex/datasets/optic_disc_seg.tar.gz'
pdx.utils.download_and_decompress(optic_dataset, path='./')

# 定义训练和验证时的transforms
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/apis/transforms/transforms.md
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

# 定义训练和验证所用的数据集
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/apis/datasets.md
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
    transforms=eval_transforms,
    shuffle=False)

# 加载模型
model = pdx.load_model('output/unet/best_model')

# Step 1/3: 分析模型各层参数在不同的剪裁比例下的敏感度
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/apis/models/semantic_segmentation.md#analyze_sensitivity
model.analyze_sensitivity(
    dataset=eval_dataset, batch_size=1, save_dir='output/unet/prune')

# Step 2/3: 根据选择的FLOPs减小比例对模型进行剪裁
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/apis/models/semantic_segmentation.md#prune
model.prune(pruned_flops=.2)

# Step 3/3: 对剪裁后的模型重新训练
# API说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/apis/models/semantic_segmentation.md#train
# 各参数介绍与调整说明：https://github.com/PaddlePaddle/PaddleX/blob/develop/dygraph/docs/parameters.md
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=4,
    eval_dataset=eval_dataset,
    learning_rate=0.01,
    save_dir='output/unet/prune')
