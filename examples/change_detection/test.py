import os
import cv2
import numpy as np
from PIL import Image

import paddlex as pdx

model_dir = "output/unet_1/best_model/"
save_dir = 'output/gt_pred'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
color = [0, 0, 0, 255, 255, 255]

model = pdx.load_model(model_dir)

with open('tiled_dataset/val_list.txt', 'r') as f:
    for line in f:
        items = line.strip().split()
        img_file_1 = os.path.join('tiled_dataset', items[0])
        img_file_2 = os.path.join('tiled_dataset', items[1])
        label_file = os.path.join('tiled_dataset', items[2])

        # 预测并可视化预测结果
        im1 = cv2.imread(img_file_1)
        im2 = cv2.imread(img_file_2)
        image = np.concatenate((im1, im2), axis=-1)
        pred = model.predict(image)
        vis_pred = pdx.seg.visualize(
            img_file_1, pred, weight=0., save_dir=None, color=color)

        # 可视化标注文件
        label = np.asarray(Image.open(label_file))
        pred = {'label_map': label}
        vis_gt = pdx.seg.visualize(
            img_file_1, pred, weight=0., save_dir=None, color=color)

        ims = cv2.hconcat([im1, im2])
        labels = cv2.hconcat([vis_gt, vis_pred])
        data = cv2.vconcat([ims, labels])
        cv2.imwrite("{}/{}".format(save_dir, items[0].split('/')[-1]), data)
