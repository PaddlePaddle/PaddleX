import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx
from paddlex.cla import transforms

# 下载和解压Imagenet果蔬分类数据集
veg_dataset = 'https://bj.bcebos.com/paddlex/datasets/mini_imagenet_veg.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

# 定义测试集的transform
test_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])

# 定义测试所用的数据集
test_dataset = pdx.datasets.ImageNet(
    data_dir='mini_imagenet_veg',
    file_list=osp.join('mini_imagenet_veg', 'test_list.txt'),
    label_list=osp.join('mini_imagenet_veg', 'labels.txt'),
    transforms=test_transforms)

# 下载和解压已训练好的MobileNetV2模型
model_file = 'https://bj.bcebos.com/paddlex/models/mini_imagenet_veg_mobilenetv2.tar.gz'
pdx.utils.download_and_decompress(model_file, path='./')

# 导入模型
model = pdx.load_model('mini_imagenet_veg_mobilenetv2')

# 可解释性可视化
save_dir = 'interpret_results'
if not osp.exists(save_dir):
    os.makedirs(save_dir)
pdx.interpret.visualize('mini_imagenet_veg/mushroom/n07734744_1106.JPEG', 
          model,
          test_dataset, 
          algo='lime',
          save_dir=save_dir)
pdx.interpret.visualize('mini_imagenet_veg/mushroom/n07734744_1106.JPEG', 
          model, 
          test_dataset, 
          algo='normlime',
          save_dir=save_dir)