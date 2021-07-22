# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
import tqdm
import numpy as np
from .prune import cal_model_size


def visualize(model, sensitivities_file, save_dir='./'):
    """将模型裁剪率和每个参数裁剪后精度损失的关系可视化。
       可视化结果纵轴为eval_metric_loss参数值，横轴为对应的模型被裁剪的比例

    Args:
        model (paddlex.cv.models): paddlex中的模型。
        sensitivities_file (str): 敏感度文件存储路径。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    program = model.test_prog
    place = model.places[0]
    fig = plt.figure()
    plt.xlabel("model prune ratio")
    plt.ylabel("evaluation loss")
    title_name = osp.split(sensitivities_file)[-1].split('.')[0]
    plt.title(title_name)
    plt.grid(linestyle='--', linewidth=0.5)
    x = list()
    y = list()
    for loss_thresh in tqdm.tqdm(list(np.arange(0.05, 1, 0.05))):
        prune_ratio = 1 - cal_model_size(
            program,
            place,
            sensitivities_file,
            eval_metric_loss=loss_thresh,
            scope=model.scope)
        x.append(prune_ratio)
        y.append(loss_thresh)
    plt.plot(x, y, color='green', linewidth=0.5, marker='o', markersize=3)
    my_x_ticks = np.arange(
        min(np.array(x)) - 0.01, max(np.array(x)) + 0.01, 0.05)
    my_y_ticks = np.arange(0.05, 1, 0.05)
    plt.xticks(my_x_ticks, rotation=15, fontsize=8)
    plt.yticks(my_y_ticks, fontsize=8)
    for a, b in zip(x, y):
        plt.text(
            a,
            b, (float('%0.3f' % a), float('%0.3f' % b)),
            ha='center',
            va='bottom',
            fontsize=8)
    plt.rcParams['savefig.dpi'] = 120
    plt.rcParams['figure.dpi'] = 150
    suffix = osp.splitext(sensitivities_file)[-1]
    plt.savefig(osp.join(save_dir, 'sensitivities.png'))
    plt.close()
    import pickle
    coor = dict(zip(x, y))
    output = open(osp.join(save_dir, 'sensitivities_xy.pkl'), 'wb')
    pickle.dump(coor, output)
