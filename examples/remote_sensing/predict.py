# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx

model_dir = 'output/deeplabv3p_mobilenetv3_large_ssld/best_model'
img_file = "dataset/JPEGImages/5.png"
save_dir = 'output/deeplabv3p_mobilenetv3_large_ssld/'

model = pdx.load_model('output/deeplabv3p_mobilenetv3_large_ssld/best_model')

# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict
pred = model.overlap_tile_predict(
    img_file=img_file, tile_size=(769, 769), pad_size=[64, 64], batch_size=32)

pdx.seg.visualize(img_file, pred, weight=0., save_dir=save_dir)
