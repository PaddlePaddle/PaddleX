# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np

import paddlex as pdx

model_dir = 'output/unet_3/best_model'
data_dir = 'google_change_det_dataset'
file_list = 'google_change_det_dataset/val_list.txt'
save_dir = 'output/unet/pred'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
color = [0, 0, 0, 255, 255, 255]

model = pdx.load_model(model_dir)

with open(file_list, 'r') as f:
    for line in f:
        items = line.strip().split()
        img_file_1 = os.path.join(data_dir, items[0])
        img_file_2 = os.path.join(data_dir, items[1])

        # 原图是tiff格式的图片，PaddleX统一使用gdal库读取
        # 因训练数据已经转换成bmp格式，故此处使用opencv读取三通道的tiff图片
        #image1 = transforms.Compose.read_img(img_file_1)
        #image2 = transforms.Compose.read_img(img_file_2)
        image1 = cv2.imread(img_file_1)
        image2 = cv2.imread(img_file_2)
        image = np.concatenate((image1, image2), axis=-1)

        # API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict
        pred = model.overlap_tile_predict(
            img_file=image,
            tile_size=(769, 769),
            pad_size=[512, 512],
            batch_size=4)

        pdx.seg.visualize(
            img_file_1, pred, weight=0., save_dir=save_dir, color=color)
