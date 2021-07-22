import paddlex as pdx
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 下载训练好的模型
url = 'https://bj.bcebos.com/paddlex/models/mobilenetv2_vegetables.tar.gz'
pdx.utils.download_and_decompress(url, path='.')

# 下载相应的训练数据集
url = 'https://bj.bcebos.com/paddlex/datasets/vegetables_cls.tar.gz'
pdx.utils.download_and_decompress(url, path='.')

# 加载模型
model = pdx.load_model('mobilenetv2_vegetables')

# 将正常模型导出为部署格式，用于对比
import time
for i in range(60):
    print('save', i)
    time.sleep(1)
    model.export_inference_model('server_mobilenet')

# 加载数据集用于量化
dataset = pdx.datasets.ImageNet(
                data_dir='vegetables_cls',
                file_list='vegetables_cls/train_list.txt',
                label_list='vegetables_cls/labels.txt',
                transforms=model.test_transforms)

# 开始量化
pdx.slim.export_quant_model(model, dataset, batch_size=4, batch_num=10, save_dir='./quant_mobilenet', cache_dir='./tmp')
