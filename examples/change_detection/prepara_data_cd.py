import os
import os.path as osp
import numpy as np
import cv2
import shutil
import random
random.seed(0)
from PIL import Image
import paddlex as pdx

# 定义训练集切分时的滑动窗口大小和步长，格式为(W, H)
train_tile_size = (1024, 1024)
train_stride = (512, 512)
# 定义验证集切分时的滑动窗口大小和步长，格式(W, H)
val_tile_size = (769, 769)
val_stride = (769, 769)
# 训练集和验证集比例
train_ratio = 0.8
val_ratio = 0.2

change_det_dataset = './change_det_data'
tiled_dataset = './tiled_dataset'
origin_dataset = './origin_dataset'
tiled_image_dir = osp.join(tiled_dataset, 'JPEGImages')
tiled_anno_dir = osp.join(tiled_dataset, 'Annotations')

if not osp.exists(tiled_image_dir):
    os.makedirs(tiled_image_dir)
if not osp.exists(tiled_anno_dir):
    os.makedirs(tiled_anno_dir)

# 划分数据集
im1_file_list = os.listdir(osp.join(change_det_dataset, 'T1'))
im2_file_list = os.listdir(osp.join(change_det_dataset, 'T2'))
label_file_list = os.listdir(osp.join(change_det_dataset, 'labels_change'))
im1_file_list = sorted(
    im1_file_list, key=lambda k: int(k.split('test')[-1].split('_')[0]))
im2_file_list = sorted(
    im2_file_list, key=lambda k: int(k.split('test')[-1].split('_')[0]))
label_file_list = sorted(
    label_file_list, key=lambda k: int(k.split('test')[-1].split('_')[0]))
file_list = list()
for im1_file, im2_file, label_file in zip(im1_file_list, im2_file_list,
                                          label_file_list):
    im1_file = osp.join(osp.join(change_det_dataset, 'T1'), im1_file)
    im2_file = osp.join(osp.join(change_det_dataset, 'T2'), im2_file)
    label_file = osp.join(
        osp.join(change_det_dataset, 'labels_change'), label_file)
    file_list.append((im1_file, im2_file, label_file))
random.shuffle(file_list)
train_num = int(len(file_list) * train_ratio)

for i, item in enumerate(file_list):
    im1_file, im2_file, label_file = item[:]
    if i < train_num:
        stride = train_stride
        tile_size = train_tile_size
    else:
        stride = val_stride
        tile_size = val_tile_size
    i += 1
    set_name = 'train' if i < train_num else 'val'
    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)
    label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
    label = label != 0
    label = label.astype(np.uint8)
    H, W, C = im1.shape
    tile_id = 1
    im1_name = osp.split(im1_file)[-1].split('.')[0]
    im2_name = osp.split(im2_file)[-1].split('.')[0]
    label_name = osp.split(label_file)[-1].split('.')[0]
    for h in range(0, H, stride[1]):
        for w in range(0, W, stride[0]):
            left = w
            upper = h
            right = min(w + tile_size[0], W)
            lower = min(h + tile_size[1], H)
            tile_im1 = im1[upper:lower, left:right, :]
            tile_im2 = im2[upper:lower, left:right, :]
            cv2.imwrite(
                osp.join(tiled_image_dir,
                         "{}_{}.bmp".format(im1_name, tile_id)), tile_im1)
            cv2.imwrite(
                osp.join(tiled_image_dir,
                         "{}_{}.bmp".format(im2_name, tile_id)), tile_im2)
            cut_label = label[upper:lower, left:right]
            cv2.imwrite(
                osp.join(tiled_anno_dir,
                         "{}_{}.png".format(label_name, tile_id)), cut_label)
            mode = 'w' if i in [0, train_num] and tile_id == 1 else 'a'
            with open(
                    osp.join(tiled_dataset, '{}_list.txt'.format(set_name)),
                    mode) as f:
                f.write(
                    "JPEGImages/{}_{}.bmp JPEGImages/{}_{}.bmp Annotations/{}_{}.png\n".
                    format(im1_name, tile_id, im2_name, tile_id, label_name,
                           tile_id))
            tile_id += 1

# 生成labels.txt
label_list = ['unchanged', 'changed']
for i, label in enumerate(label_list):
    mode = 'w' if i == 0 else 'a'
    with open(osp.join(tiled_dataset, 'labels.txt'), 'a') as f:
        name = "{}\n".format(label) if i < len(
            label_list) - 1 else "{}".format(label)
        f.write(name)
