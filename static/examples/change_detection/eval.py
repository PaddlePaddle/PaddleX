# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict

import paddlex as pdx
import paddlex.utils.logging as logging
from paddlex.seg import transforms
from paddlex.cv.models.utils.seg_eval import ConfusionMatrix

model_dir = 'output/unet/best_model'
data_dir = 'google_change_det_dataset'
file_list = 'google_change_det_dataset/val_list.txt'


def update_confusion_matrix(confusion_matrix, predction, label):
    pred = predction["label_map"]
    pred = pred[np.newaxis, :, :, np.newaxis]
    pred = pred.astype(np.int64)
    label = label[np.newaxis, np.newaxis, :, :]
    mask = label != model.ignore_index
    confusion_matrix.calculate(pred=pred, label=label, ignore=mask)


model = pdx.load_model(model_dir)

conf_mat = ConfusionMatrix(model.num_classes, streaming=True)

with open(file_list, 'r') as f:
    for line in f:
        items = line.strip().split()
        full_path_im1 = osp.join(data_dir, items[0])
        full_path_im2 = osp.join(data_dir, items[1])
        full_path_label = osp.join(data_dir, items[2])

        # 原图是tiff格式的图片，PaddleX统一使用gdal库读取
        # 因训练数据已经转换成bmp格式，故此处使用opencv读取三通道的tiff图片
        #image1 = transforms.Compose.read_img(full_path_im1)
        #image2 = transforms.Compose.read_img(full_path_im2)
        image1 = cv2.imread(full_path_im1)
        image2 = cv2.imread(full_path_im2)
        image = np.concatenate((image1, image2), axis=-1)

        # API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict
        overlap_tile_predict = model.overlap_tile_predict(
            img_file=image,
            tile_size=(769, 769),
            pad_size=[512, 512],
            batch_size=4)

        # 将三通道的label图像转换成单通道的png格式图片
        # 且将标注0和255转换成0和1
        label = cv2.imread(full_path_label)
        label = label[:, :, 0]
        label = label != 0
        label = label.astype(np.uint8)
        update_confusion_matrix(conf_mat, overlap_tile_predict, label)

category_iou, miou = conf_mat.mean_iou()
category_acc, oacc = conf_mat.accuracy()
category_f1score = conf_mat.f1_score()

logging.info(
    "miou={:.6f} category_iou={} oacc={:.6f} category_acc={} kappa={:.6f} category_F1-score={}".
    format(miou, category_iou, oacc, category_acc,
           conf_mat.kappa(), conf_mat.f1_score()))
