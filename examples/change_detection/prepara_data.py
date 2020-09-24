import os
import os.path as osp
import numpy as np
import cv2
import shutil
from PIL import Image
import paddlex as pdx

# 定义训练集切分时的滑动窗口大小和步长，格式为(W, H)
train_tile_size = (512, 512)
train_stride = (256, 256)
# 定义验证集切分时的滑动窗口大小和步长，格式(W, H)
val_tile_size = (256, 256)
val_stride = (256, 256)

## 下载并解压2015 CCF大数据比赛提供的高清遥感影像
#SZTAKI_AirChange_Benchmark = 'https://bj.bcebos.com/paddlex/examples/remote_sensing/datasets/ccf_remote_dataset.tar.gz'
#pdx.utils.download_and_decompress(SZTAKI_AirChange_Benchmark, path='./')

if not osp.exists('./dataset/JPEGImages'):
    os.makedirs('./dataset/JPEGImages')
if not osp.exists('./dataset/Annotations'):
    os.makedirs('./dataset/Annotations')

# 将前4张图片划分入训练集，并切分成小块之后加入到训练集中
# 并生成train_list.txt
train_list = {'Szada': [2, 3, 4, 5, 6, 7], 'Tiszadob': [1, 2, 4, 5]}
val_list = {'Szada': [1], 'Tiszadob': [3]}
all_list = [train_list, val_list]

for i, data_list in enumerate(all_list):
    id = 0
    if i == 0:
        for key, value in data_list.items():
            for v in value:
                shutil.copyfile(
                    "SZTAKI_AirChange_Benchmark/{}/{}/im1.bmp".format(key, v),
                    "./dataset/JPEGImages/{}_{}_im1.bmp".format(key, v))
                shutil.copyfile(
                    "SZTAKI_AirChange_Benchmark/{}/{}/im2.bmp".format(key, v),
                    "./dataset/JPEGImages/{}_{}_im2.bmp".format(key, v))
                label = cv2.imread(
                    "SZTAKI_AirChange_Benchmark/{}/{}/gt.bmp".format(key, v))
                label = label[:, :, 0]
                label = label != 0
                label = label.astype(np.uint8)
                cv2.imwrite("./dataset/Annotations/{}_{}_gt.png".format(
                    key, v), label)

                id += 1
                mode = 'w' if id == 1 else 'a'
                with open('./dataset/train_list.txt', mode) as f:
                    f.write(
                        "JPEGImages/{}_{}_im1.bmp JPEGImages/{}_{}_im2.bmp Annotations/{}_{}_gt.png\n".
                        format(key, v, key, v, key, v))

    if i == 0:
        stride = train_stride
        tile_size = train_tile_size
    else:
        stride = val_stride
        tile_size = val_tile_size
    for key, value in data_list.items():
        for v in value:
            im1 = cv2.imread("SZTAKI_AirChange_Benchmark/{}/{}/im1.bmp".format(
                key, v))
            im2 = cv2.imread("SZTAKI_AirChange_Benchmark/{}/{}/im2.bmp".format(
                key, v))
            label = cv2.imread(
                "SZTAKI_AirChange_Benchmark/{}/{}/gt.bmp".format(key, v))
            label = label[:, :, 0]
            label = label != 0
            label = label.astype(np.uint8)
            H, W, C = im1.shape
            tile_id = 1
            for h in range(0, H, stride[1]):
                for w in range(0, W, stride[0]):
                    left = w
                    upper = h
                    right = min(w + tile_size[0], W)
                    lower = min(h + tile_size[1], H)
                    tile_im1 = im1[upper:lower, left:right, :]
                    tile_im2 = im2[upper:lower, left:right, :]
                    cv2.imwrite("./dataset/JPEGImages/{}_{}_{}_im1.bmp".format(
                        key, v, tile_id), tile_im1)
                    cv2.imwrite("./dataset/JPEGImages/{}_{}_{}_im2.bmp".format(
                        key, v, tile_id), tile_im2)
                    cut_label = label[upper:lower, left:right]
                    cv2.imwrite("./dataset/Annotations/{}_{}_{}_gt.png".format(
                        key, v, tile_id), cut_label)
                    with open('./dataset/{}_list.txt'.format(
                            'train' if i == 0 else 'val'), 'a') as f:
                        f.write(
                            "JPEGImages/{}_{}_{}_im1.bmp JPEGImages/{}_{}_{}_im2.bmp Annotations/{}_{}_{}_gt.png\n".
                            format(key, v, tile_id, key, v, tile_id, key, v,
                                   tile_id))
                    tile_id += 1

# 生成labels.txt
label_list = ['unchanged', 'changed']
for i, label in enumerate(label_list):
    mode = 'w' if i == 0 else 'a'
    with open('./dataset/labels.txt', 'a') as f:
        name = "{}\n".format(label) if i < len(
            label_list) - 1 else "{}".format(label)
        f.write(name)
