import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex import transforms

veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224), transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])

eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224), transforms.Normalize()
])

train_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/train_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.ImageNet(
    data_dir='vegetables_cls',
    file_list='vegetables_cls/val_list.txt',
    label_list='vegetables_cls/labels.txt',
    transforms=eval_transforms)

model = pdx.load_model('output/mobilenet_v2/best_model')

# Step 1/3: Analyze the sensitivities of parameters
model.analyze_sensitivity(
    dataset=eval_dataset, save_dir='output/mobilenet_v2/prune')

# Step 2/3: Prune the model by the specified ratio of FLOPs to be pruned
model.prune(pruned_flops=.2)

# Step 3/3: Retrain the model
model.train(
    num_epochs=10,
    train_dataset=train_dataset,
    train_batch_size=32,
    eval_dataset=eval_dataset,
    lr_decay_epochs=[4, 6, 8],
    learning_rate=0.025,
    pretrain_weights=None,
    save_dir='output/mobilenet_v2/prune',
    use_vdl=True)
