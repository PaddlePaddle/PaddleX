# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""

import os
import json
import math
import platform
from pathlib import Path

from collections import defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .....utils.file_interface import custom_open
from ....utils.fonts import PINGFANG_FONT_FILE_PATH


def deep_analyse(dataset_path, output_dir):
    """class analysis for dataset"""
    tags = ['train', 'val', 'test']
    labels_cnt = defaultdict(str)
    label_path = os.path.join(dataset_path, 'label.txt')
    with custom_open(label_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        labels_cnt[line[0]] = " ".join(line[1:])
    for tag in tags:
        image_path = os.path.join(dataset_path, f'{tag}.txt')
        if tag == 'test' and not os.path.exists(image_path):
            cnts_test = None
            continue
        str_nums = []
        classes_num = defaultdict(int)
        for i in range(len(labels_cnt)):
            classes_num[labels_cnt[str(i)]] = 0
        with custom_open(image_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            classes_num[labels_cnt[line[1]]] += 1
        if tag == 'train':
            cnts_train = [cat_ids for cat_name, cat_ids in classes_num.items()]
        elif tag == 'val':
            cnts_val = [cat_ids for cat_name, cat_ids in classes_num.items()]
        else:
            cnts_test = [cat_ids for cat_name, cat_ids in classes_num.items()]

    classes = [cat_name for cat_name, cat_ids in classes_num.items()]
    sorted_id = sorted(
        range(len(cnts_train)), key=lambda k: cnts_train[k], reverse=True)
    cnts_train_sorted = [cnts_train[index] for index in sorted_id]
    cnts_val_sorted = [cnts_val[index] for index in sorted_id]
    if cnts_test:
        cnts_test_sorted = [cnts_test[index] for index in sorted_id]
    classes_sorted = [classes[index] for index in sorted_id]
    x = np.arange(len(classes))
    width = 0.5 if not cnts_test else 0.333

    # bar
    os_system = platform.system().lower()
    if os_system == "windows":
        plt.rcParams['font.sans-serif'] = 'FangSong'
    else:
        font = font_manager.FontProperties(
            fname=PINGFANG_FONT_FILE_PATH, size=10)
    fig, ax = plt.subplots(figsize=(max(8, int(len(classes) / 5)), 5), dpi=300)
    ax.bar(x,
           cnts_train_sorted,
           width=0.5 if not cnts_test else 0.333,
           label='train')
    ax.bar(x + width,
           cnts_val_sorted,
           width=0.5 if not cnts_test else 0.333,
           label='val')
    if cnts_test:
        ax.bar(x + 2 * width, cnts_test_sorted, width=0.333, label='test')
    plt.xticks(
        x + width / 2 if not cnts_test else x + width,
        classes_sorted,
        rotation=90,
        fontproperties=None if os_system == "windows" else font)
    ax.set_xlabel(
        '类别名称',
        fontproperties=None if os_system == "windows" else font,
        fontsize=12)
    ax.set_ylabel(
        '图片数量',
        fontproperties=None if os_system == "windows" else font,
        fontsize=12)
    plt.legend(loc=1)
    fig.tight_layout()
    file_path = os.path.join(output_dir, "histogram.png")
    fig.savefig(file_path, dpi=300)

    return {"histogram": "histogram.png"}
