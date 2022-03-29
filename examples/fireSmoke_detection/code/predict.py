# coding:utf-8
# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as osp
import paddlex as pdx

img_file = 'images/000001.jpg'
model_dir = 'output/ppyolov2_r50vd_dcn/best_model/'
save_dir = './visualize/predict'
# 设置置信度阈值
score_threshold = 0.4

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = pdx.load_model(model_dir)
res = model.predict(img_file)
pdx.det.visualize(img_file, res, threshold=score_threshold, save_dir=save_dir)
