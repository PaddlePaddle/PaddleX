# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import cv2
from PIL import Image
from collections import OrderedDict

import paddlex as pdx
import paddlex.utils.logging as logging
from paddlex.cv.models.utils.seg_eval import ConfusionMatrix


def update_confusion_matrix(confusion_matrix, predction, label):
    pred = predction["label_map"]
    pred = pred[np.newaxis, :, :, np.newaxis]
    pred = pred.astype(np.int64)
    label = label[np.newaxis, np.newaxis, :, :]
    mask = label != model.ignore_index
    confusion_matrix.calculate(pred=pred, label=label, ignore=mask)


model_dir = 'output/deeplabv3p_mobilenetv3_large_ssld/best_model'
img_file = "dataset/JPEGImages/5.png"
label_file = "dataset/Annotations/5_class.png"

model = pdx.load_model(model_dir)

conf_mat = ConfusionMatrix(model.num_classes, streaming=True)

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict
overlap_tile_predict = model.overlap_tile_predict(
    img_file=img_file, tile_size=(769, 769), pad_size=[64, 64], batch_size=32)

label = np.asarray(Image.open(label_file))
update_confusion_matrix(conf_mat, overlap_tile_predict, label)

category_iou, miou = conf_mat.mean_iou()
category_acc, macc = conf_mat.accuracy()
logging.info(
    "miou={:.6f} category_iou={} macc={:.6f} category_acc={} kappa={:.6f}".
    format(miou, category_iou, macc, category_acc, conf_mat.kappa()))
