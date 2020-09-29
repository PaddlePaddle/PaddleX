import os
import os.path as osp
import numpy as np
import cv2
import shutil
import random
# 为保证每次运行该脚本时划分的样本一致，故固定随机种子
random.seed(0)

import paddlex as pdx

# 定义训练集切分时的滑动窗口大小和步长，格式为(W, H)
train_tile_size = (1024, 1024)
train_stride = (512, 512)
# 定义验证集切分时的滑动窗口大小和步长，格式(W, H)
val_tile_size = (769, 769)
val_stride = (769, 769)
# 训练集和验证集比例
train_ratio = 0.75
val_ratio = 0.25
# 切分后的数据集保存路径
tiled_dataset = './tiled_dataset'
# 切分后的图像文件保存路径
tiled_image_dir = osp.join(tiled_dataset, 'JPEGImages')
# 切分后的标注文件保存路径
tiled_anno_dir = osp.join(tiled_dataset, 'Annotations')

# 下载和解压Google Dataset数据集
change_det_dataset = 'https://bj.bcebos.com/paddlex/examples/change_detection/google_change_det_dataset.tar.gz'
pdx.utils.download_and_decompress(change_det_dataset, path='./')
change_det_dataset = './google_change_det_dataset'
image1_dir = osp.join(change_det_dataset, 'T1')
image2_dir = osp.join(change_det_dataset, 'T2')
label_dir = osp.join(change_det_dataset, 'labels_change')

if not osp.exists(tiled_image_dir):
    os.makedirs(tiled_image_dir)
if not osp.exists(tiled_anno_dir):
    os.makedirs(tiled_anno_dir)

# 划分数据集
im1_file_list = os.listdir(image1_dir)
im2_file_list = os.listdir(image2_dir)
label_file_list = os.listdir(label_dir)
im1_file_list = sorted(
    im1_file_list, key=lambda k: int(k.split('test')[-1].split('_')[0]))
im2_file_list = sorted(
    im2_file_list, key=lambda k: int(k.split('test')[-1].split('_')[0]))
label_file_list = sorted(
    label_file_list, key=lambda k: int(k.split('test')[-1].split('_')[0]))

file_list = list()
for im1_file, im2_file, label_file in zip(im1_file_list, im2_file_list,
                                          label_file_list):
    im1_file = osp.join(image1_dir, im1_file)
    im2_file = osp.join(image2_dir, im2_file)
    label_file = osp.join(label_dir, label_file)
    file_list.append((im1_file, im2_file, label_file))
random.shuffle(file_list)
train_num = int(len(file_list) * train_ratio)

# 将大图切分成小图
for i, item in enumerate(file_list):
    if i < train_num:
        stride = train_stride
        tile_size = train_tile_size
    else:
        stride = val_stride
        tile_size = val_tile_size
    set_name = 'train' if i < train_num else 'val'

    # 生成原图的file_list
    im1_file, im2_file, label_file = item[:]
    mode = 'w' if i in [0, train_num] else 'a'
    with open(
            osp.join(change_det_dataset, '{}_list.txt'.format(set_name)),
            mode) as f:
        f.write("T1/{} T2/{} labels_change/{}\n".format(
            osp.split(im1_file)[-1],
            osp.split(im2_file)[-1], osp.split(label_file)[-1]))

    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)
    # 将三通道的label图像转换成单通道的png格式图片
    # 且将标注0和255转换成0和1
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
