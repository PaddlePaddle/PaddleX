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

# 导入模型参数
model = pdx.load_model('output/deeplabv3p_mobilenetv3_large_ssld/best_model')

# 指定待评估图像路径及其标注文件路径
img_file = "dataset/JPEGImages/5.png"
label_file = "dataset/Annotations/5_class.png"

# 定义用于计算miou、iou、macc、acc、kapp指标的混淆矩阵类
conf_mat = ConfusionMatrix(model.num_classes, streaming=True)

# 使用"无重叠的大图切小图"方式进行预测：将大图像切分成互不重叠多个小块，分别对每个小块进行预测
# 最后将小块预测结果拼接成大图预测结果
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#tile-predict
# tile_predict = model.tile_predict(img_file=img_file, tile_size=(769, 769))
# pred = tile_predict["label_map"]

# 使用"有重叠的大图切小图"策略进行预测：将大图像切分成相互重叠的多个小块，
# 分别对每个小块进行预测，将小块预测结果的中间部分拼接成大图预测结果
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict
overlap_tile_predict = model.overlap_tile_predict(
    img_file=img_file, tile_size=(769, 769))
pred = overlap_tile_predict["label_map"]

# 更新混淆矩阵
pred = pred[np.newaxis, :, :, np.newaxis]
pred = pred.astype(np.int64)
label = np.asarray(Image.open("dataset/Annotations/5_class.png"))
label = label[np.newaxis, np.newaxis, :, :]
mask = label != model.ignore_index
conf_mat.calculate(pred=pred, label=label, ignore=mask)

# 计算miou、iou、macc、acc、kapp
category_iou, miou = conf_mat.mean_iou()
category_acc, macc = conf_mat.accuracy()
logging.info(
    "miou={:.6f} category_iou={} macc={:.6f} category_acc={} kappa={:.6f}".
    format(miou, category_iou, macc, category_acc, conf_mat.kappa()))
