#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_segmented_result(save_path, image_groundtruth, groundtruth,
                               image_predict, predict, legend):
    tail = save_path.split(".")[-1]
    save_path = (save_path[:-len(tail)] + "png")
    import matplotlib.patches as mpatches
    from matplotlib import use
    use('Agg')
    if image_groundtruth is not None:
        image_groundtruth = image_groundtruth[..., ::-1]
    image_predict = image_predict[..., ::-1]
    if groundtruth is not None:
        groundtruth = groundtruth[..., ::-1]
    predict = predict[..., ::-1]
    fig = plt.figure()
    red_patches = []
    for key, value in legend.items():
        red_patch = mpatches.Patch(
            color=[x / 255.0 for x in value[::-1]], label=key)
        red_patches.append(red_patch)
    plt.legend(
        handles=red_patches, bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.axis('off')

    if image_groundtruth is not None and \
            groundtruth is not None:
        left, bottom, width, height = 0.02, 0.51, 0.38, 0.38
        fig.add_axes([left, bottom, width, height])
        plt.imshow(image_groundtruth)
        plt.axis('off')
        plt.title("Ground Truth", loc='left')
        left, bottom, width, height = 0.52, 0.51, 0.38, 0.38
        fig.add_axes([left, bottom, width, height])
        plt.imshow(groundtruth)
        plt.axis('off')
        left, bottom, width, height = 0.01, 0.5, 0.9, 0.45
        fig.add_axes([left, bottom, width, height])
        currentAxis = plt.gca()
        rect = patches.Rectangle(
            (0.0, 0.0), 1.0, 1.0, linewidth=1, edgecolor='k', facecolor='none')
        currentAxis.add_patch(rect)
        plt.axis('off')

        left, bottom, width, height = 0.02, 0.06, 0.38, 0.38
        fig.add_axes([left, bottom, width, height])
        plt.imshow(image_predict)
        plt.axis('off')
        plt.title("Prediction", loc='left')
        left, bottom, width, height = 0.52, 0.06, 0.38, 0.38
        fig.add_axes([left, bottom, width, height])
        plt.imshow(predict)
        plt.axis('off')
        left, bottom, width, height = 0.01, 0.05, 0.9, 0.45
        fig.add_axes([left, bottom, width, height])
        currentAxis = plt.gca()
        rect = patches.Rectangle(
            (0.0, 0.0), 1.0, 1.0, linewidth=1, edgecolor='k', facecolor='none')
        currentAxis.add_patch(rect)
        plt.axis('off')
    else:
        plt.subplot(1, 2, 1)
        plt.imshow(image_predict)
        plt.axis('off')
        plt.title("Combination ", y=-0.12)
        plt.subplot(1, 2, 2)
        plt.imshow(predict)
        plt.axis('off')
        plt.title("Prediction", y=-0.12)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_detected_result(save_path, image_groundtruth, image_predict):
    tail = save_path.split(".")[-1]
    save_path = (save_path[:-len(tail)] + "png")
    from matplotlib import use
    use('Agg')
    if image_groundtruth is not None:
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image_groundtruth, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Ground Truth", y=-0.12)
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image_predict, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Prediction", y=-0.12)
    else:
        plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(image_predict, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Prediction", y=-0.12)
    plt.tight_layout(pad=1.08)
    plt.autoscale()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def visualize_classified_result(save_path, image_predict, res_info):
    from matplotlib import use
    use('Agg')
    if isinstance(image_predict, str):
        img = Image.open(image_predict)
        name_part = osp.split(image_predict)
        filename = name_part[-1]
        foldername = osp.split(name_part[-2])[-1]
        tail = filename.split(".")[-1]
        filename = (foldername + '_' + filename[:-len(tail)] + "png")
    elif isinstance(image_predict, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image_predict, cv2.COLOR_BGR2RGB))
        filename = "predict_result.png"
    if np.array(img).ndim == 3:
        cmap = None
    else:
        cmap = 'gray'
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    if "gt_label" in res_info:
        plt.title(
            "Test Image, Label: {}".format(res_info["gt_label"]), y=-0.15)
    else:
        plt.title("Test Image", y=-0.15)
    plt.subplot(1, 2, 2)
    topk = res_info["topk"]
    start_height = (topk + 2) // 2 * 10 + 45
    plt.text(
        15, start_height, 'Probability of each class:', va='center', ha='left')
    for i in range(topk):
        if "gt_label" in res_info:
            color = "red" if res_info["label"][i] == res_info[
                "gt_label"] else "black"
        else:
            color = 'black'
        if i == 0:
            color = "green"
        plt.text(
            70,
            start_height - (i + 1) * 10,
            '    {}: {:.4f}'.format(res_info["label"][i],
                                    res_info["score"][i]),
            va='center',
            ha='right',
            color=color)
    if "gt_label" in res_info:
        plt.text(
            15,
            start_height - (topk + 1) * 10,
            'True Label: {}'.format(res_info["gt_label"]),
            va='center',
            ha='left',
            color="black")
    plt.axis('off')
    plt.axis([0, 100, 0, 100])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout(pad=0.08)
    plt.savefig(osp.join(save_path, filename), dpi=200, bbox_inches='tight')
    plt.close()
