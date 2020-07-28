import paddlex as pdx

# 下载和解压Imagenet果蔬分类数据集
veg_dataset = 'https://bj.bcebos.com/paddlex/interpret/mini_imagenet_veg.tar.gz'
pdx.utils.download_and_decompress(veg_dataset, path='./')

# 下载和解压已训练好的MobileNetV2模型
model_file = 'https://bj.bcebos.com/paddlex/interpret/mini_imagenet_veg_mobilenetv2.tar.gz'
pdx.utils.download_and_decompress(model_file, path='./')

# 加载模型
model = pdx.load_model('mini_imagenet_veg_mobilenetv2')

# 可解释性可视化
pdx.interpret.lime(
    'mini_imagenet_veg/mushroom/n07734744_1106.JPEG', model, save_dir='./')
