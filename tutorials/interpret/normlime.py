import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as osp
import paddlex as pdx

# 下载和解压Imagenet果蔬分类数据集
veg_dataset = 'https://bj.bcebos.com/paddlex/interpret/mini_imagenet_veg.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

# 下载和解压已训练好的MobileNetV2模型
model_file = 'https://bj.bcebos.com/paddlex/interpret/mini_imagenet_veg_mobilenetv2.tar.gz'
pdx.utils.download_and_decompress(model_file, path='./')

# 加载模型
model_file = 'mini_imagenet_veg_mobilenetv2'
model = pdx.load_model(model_file)

# 定义测试所用的数据集
dataset = 'mini_imagenet_veg'
test_dataset = pdx.datasets.ImageNet(
    data_dir=dataset,
    file_list=osp.join(dataset, 'test_list.txt'),
    label_list=osp.join(dataset, 'labels.txt'),
    transforms=model.test_transforms)

import numpy as np
perm = np.random.permutation(len(test_dataset.file_list))

for i in range(len(test_dataset.file_list)):

    # 可解释性可视化
    pdx.interpret.normlime(
        test_dataset.file_list[perm[i]][0],
        model,
        test_dataset,
        num_samples=3000,
        save_dir='./',
        normlime_weights_file='{}_{}.npy'.format(
            dataset.split('/')[-1], model.model_name))

    if i == 1:
        # first iter will have an initialization process, followed by the interpretation.
        # second iter will directly load the initialization process, followed by the interpretation.
        break
