# coding: utf8
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 环境变量配置，用于控制是否使用GPU
# 说明文档：https://paddlex.readthedocs.io/zh_CN/develop/appendix/parameters.html#gpu
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as osp
import paddlex as pdx

img_file = 'aluminum_inspection/JPEGImages/budaodian-116.jpg'
model_dir = 'output/faster_rcnn_r50_vd_dcn/best_model/'
save_dir = './visualize/predict'
# 设置置信度阈值
score_threshold = 0.9

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model = pdx.load_model(model_dir)
res = model.predict(img_file)
pdx.det.visualize(img_file, res, threshold=score_threshold, save_dir=save_dir)
