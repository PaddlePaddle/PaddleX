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
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as osp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import paddlex as pdx


def cal_image_level_recall_rate(model, dataset_dir):
    """计算置信度（Score）在[0, 1]内以间隔0.01递增取值时，模型在有目标的图像上的图片级召回率。

    图片级召回率：只要在有目标的图片上检测出目标（不论框的个数），该图片被认为召回，
       批量有目标图片中被召回的图片所占的比例，即为图片级别的召回率。

    Args:
        model (PaddleX model object): 已加载的PaddleX模型。
        dataset_dir (str)：数据集路径。

    Returns:
        numpy.array: 形状为101x1的数组，对应置信度从0到1按0.01递增取值时，计算所得图片级别的召回率。
    """

    print(
        "Begin to calculate image-level recall rate of positive images. Please wait for a moment..."
    )
    file_list = osp.join(dataset_dir, 'val_list.txt')
    tp = np.zeros((101, 1))
    positive_num = 0
    with open(file_list, 'r') as fr:
        while True:
            line = fr.readline()
            if not line:
                break
            img_file, xml_file = [osp.join(dataset_dir, x) \
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
    return tp


def cal_image_level_false_positive_rate(model, negative_dir):
    """计算置信度（Score）在[0, 1]内以间隔0.01递增取值时，模型在无目标的图像上的图片级误检率。

    图片级误检率：只要在无目标的图片上检测出目标（不论框的个数），该图片被认为误检，
       批量无目标图片中被误检的图片所占的比例，即为图片级别的误检率。

    Args:
        model (PaddleX model object): 已加载的PaddleX模型。
        negative_dir (str)：无目标图片的文件夹路径。

    Returns:
        numpy.array: 形状为101x1的数组，对应置信度从0到1按0.01递增取值时，计算所得图片级别的误检率。
    """

    print(
        "Begin to calculate image-level false positive rate of negative(background) images. Please wait for a moment..."
    )
    fp = np.zeros((101, 1))
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
    return fp


def result2textfile(tp_list, fp_list, save_dir):
    """将不同置信度阈值下的图片级召回率和图片级误检率保存为文本文件。

    文本文件中内容按照| 置信度阈值 | 图片级召回率 | 图片级误检率 |的格式保存。

    Args:
        tp_list (numpy.array): 形状为101x1的数组，对应置信度从0到1按0.01递增取值时，计算所得图片级别的召回率。
        fp_list (numpy.array): 形状为101x1的数组，对应置信度从0到1按0.01递增取值时，计算所得图片级别的误检率。
        save_dir (str): 文本文件的保存路径。

    """

    tp_fp_list_file = osp.join(save_dir, 'tp_fp_list.txt')
    with open(tp_fp_list_file, 'w') as f:
        f.write("| score | recall rate | false-positive rate |\n")
        f.write("| -- | -- | -- |\n")
        for i in range(100):
            f.write("| {:2f} | {:2f} | {:2f} |\n".format(0.01 * i, tp_list[
                i, 0], fp_list[i, 0]))
    print("The numerical score-recall_rate-false_positive_rate is saved as {}".
          format(tp_fp_list_file))


def result2imagefile(tp_list, fp_list, save_dir):
    """将不同置信度阈值下的图片级召回率和图片级误检率保存为图片。

    图片中左子图横坐标表示不同置信度阈值下计算得到的图片级召回率，纵坐标表示各图片级召回率对应的图片级误检率。
        右边子图横坐标表示图片级召回率，纵坐标表示各图片级召回率对应的置信度阈值。

    Args:
        tp_list (numpy.array): 形状为101x1的数组，对应置信度从0到1按0.01递增取值时，计算所得图片级别的召回率。
        fp_list (numpy.array): 形状为101x1的数组，对应置信度从0到1按0.01递增取值时，计算所得图片级别的误检率。
        save_dir (str): 文本文件的保存路径。

    """

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
    plt.plot(tp_list, fp_list, color='b', label="image level", linewidth=1)
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
        tp_list,
        np.arange(0, 1.01, 0.01),
        color='b',
        label="image level",
        linewidth=1)
    plt.legend(loc="lower left", fontsize=5)
    tp_fp_chart_file = os.path.join(save_dir, "image-level_tp_fp.png")
    plt.savefig(tp_fp_chart_file, dpi=800)
    plt.close()
    print(
        "The diagrammatic score-recall_rate-false_positive_rate is saved as {}".
        format(tp_fp_chart_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        default="./output/faster_rcnn_r50_vd_dcn/best_model/",
        type=str,
        help="The model directory path.")
    parser.add_argument(
        "--dataset_dir",
        default="./aluminum_inspection",
        type=str,
        help="The VOC-format dataset directory path.")
    parser.add_argument(
        "--background_image_dir",
        default="./aluminum_inspection/val_wu_xia_ci",
        type=str,
        help="The directory path of background images.")
    parser.add_argument(
        "--save_dir",
        default="./visualize/faster_rcnn_r50_vd_dcn",
        type=str,
        help="The directory path of result.")

    args = parser.parse_args()

    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = pdx.load_model(args.model_dir)

    tp_list = cal_image_level_recall_rate(model, args.dataset_dir)
    fp_list = cal_image_level_false_positive_rate(model,
                                                  args.background_image_dir)
    result2textfile(tp_list, fp_list, args.save_dir)
    result2imagefile(tp_list, fp_list, args.save_dir)
