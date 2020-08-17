# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import paddlex as pdx

# 导入模型参数
model = pdx.load_model('output/deeplabv3p_mobilenetv3_large_ssld/best_model')

# 指定待预测图像路径
img_file = "dataset/JPEGImages/5.png"

# 使用"无重叠的大图切小图"方式进行预测：将大图像切分成互不重叠多个小块，分别对每个小块进行预测
# 最后将小块预测结果拼接成大图预测结果
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#tile-predict
# pred = model.tile_predict(img_file=img_file, tile_size=(769, 769))

# 使用"有重叠的大图切小图"策略进行预测：将大图像切分成相互重叠的多个小块，
# 分别对每个小块进行预测，将小块预测结果的中间部分拼接成大图预测结果
# API说明：https://paddlex.readthedocs.io/zh_CN/develop/apis/models/semantic_segmentation.html#overlap-tile-predict
pred = model.overlap_tile_predict(img_file=img_file, tile_size=(769, 769))

# 可视化预测结果
# API说明：
pdx.seg.visualize(
    img_file,
    pred,
    weight=0.,
    save_dir='output/deeplabv3p_mobilenetv3_large_ssld/')
