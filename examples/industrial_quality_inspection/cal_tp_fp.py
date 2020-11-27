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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paddlex as pdx

data_dir = 'aluminum_inspection/'
positive_file_list = 'aluminum_inspection/val_list.txt'
negative_dir = 'aluminum_inspection/val_wu_xia_ci'
model_dir = 'output/faster_rcnn_r50_vd_dcn/best_model/'
save_dir = 'visualize/faster_rcnn_r50_vd_dcn'
if not osp.exists(save_dir):
    os.makedirs(save_dir)

tp = np.zeros((101, 1))
fp = np.zeros((101, 1))

# 导入模型
model = pdx.load_model(model_dir)

# 计算图片级召回率
print(
    "Begin to calculate image-level recall rate of positive images. Please wait for a moment..."
)
positive_num = 0
with open(positive_file_list, 'r') as fr:
    while True:
        line = fr.readline()
        if not line:
            break
        img_file, xml_file = [osp.join(data_dir, x) \
                for x in line.strip().split()[:2]]
        if not osp.exists(img_file):
            continue
        if not osp.exists(xml_file):
            continue

        positive_num += 1
        results = model.predict(img_file)
        scores = list()
        for res in results:
            scores.append(res['score'])
        if len(scores) > 0:
            tp[0:int(np.round(max(scores) / 0.01)), 0] += 1
tp = tp / positive_num

# 计算图片级误检率
print(
    "Begin to calculate image-level false-positive rate of background images. Please wait for a moment..."
)
negative_num = 0
for file in os.listdir(negative_dir):
    file = osp.join(negative_dir, file)
    results = model.predict(file)
    negative_num += 1
    scores = list()
    for res in results:
        scores.append(res['score'])
    if len(scores) > 0:
        fp[0:int(np.round(max(scores) / 0.01)), 0] += 1
fp = fp / negative_num

# 保存结果
tp_fp_list_file = osp.join(save_dir, 'tp_fp_list.txt')
with open(tp_fp_list_file, 'w') as f:
    f.write("| score | recall rate | false-positive rate |\n")
    f.write("| -- | -- | -- |\n")
    for i in range(100):
        f.write("| {:2f} | {:2f} | {:2f} |\n".format(0.01 * i, tp[i, 0], fp[
            i, 0]))
print("The numerical score-recall_rate-false_positive_rate is saved as {}".
      format(tp_fp_list_file))

plt.subplot(1, 2, 1)
plt.title("image-level false_positive-recall")
plt.xlabel("recall")
plt.ylabel("false_positive")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(linestyle='--', linewidth=1)
plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
my_x_ticks = np.arange(0, 1, 0.1)
my_y_ticks = np.arange(0, 1, 0.1)
plt.xticks(my_x_ticks, fontsize=5)
plt.yticks(my_y_ticks, fontsize=5)
plt.plot(tp, fp, color='b', label="image level", linewidth=1)
plt.legend(loc="lower left", fontsize=5)

plt.subplot(1, 2, 2)
plt.title("score-recall")
plt.xlabel('recall')
plt.ylabel('score')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(linestyle='--', linewidth=1)
plt.xticks(my_x_ticks, fontsize=5)
plt.yticks(my_y_ticks, fontsize=5)
plt.plot(
    tp, np.arange(0, 1.01, 0.01), color='b', label="image level", linewidth=1)
plt.legend(loc="lower left", fontsize=5)
tp_fp_chart_file = os.path.join(save_dir, "image-level_tp_fp.png")
plt.savefig(tp_fp_chart_file, dpi=800)
plt.close()
print("The diagrammatic score-recall_rate-false_positive_rate is saved as {}".
      format(tp_fp_chart_file))
