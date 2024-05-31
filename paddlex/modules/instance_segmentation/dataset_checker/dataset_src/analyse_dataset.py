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
import platform
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from pycocotools.coco import COCO

from ....utils.fonts import PINGFANG_FONT_FILE_PATH


def deep_analyse(dataset_dir, output_dir):
    """class analysis for dataset"""
    tags = ['train', 'val', 'test']
    all_instances = 0
    for tag in tags:
        annotations_path = os.path.abspath(
            os.path.join(dataset_dir, f'annotations/instance_{tag}.json'))
        if tag == 'test' and not os.path.exists(annotations_path):
            cnts_test = None
            continue
        labels_cnt = defaultdict(list)
        coco = COCO(annotations_path)
        cat_ids = coco.getCatIds()
        for cat_id in cat_ids:
            cat_name = coco.loadCats(ids=cat_id)[0]["name"]
            labels_cnt[cat_name] = labels_cnt[cat_name] + coco.getAnnIds(
                catIds=cat_id)
            all_instances += len(labels_cnt[cat_name])
        if tag == 'train':
            cnts_train = [
                len(cat_ids) for cat_name, cat_ids in labels_cnt.items()
            ]
        elif tag == 'val':
            cnts_val = [
                len(cat_ids) for cat_name, cat_ids in labels_cnt.items()
            ]
        else:
            cnts_test = [
                len(cat_ids) for cat_name, cat_ids in labels_cnt.items()
            ]
    classes = [cat_name for cat_name, cat_ids in labels_cnt.items()]
    sorted_id = sorted(
        range(len(cnts_train)), key=lambda k: cnts_train[k], reverse=True)
    cnts_train_sorted = sorted(cnts_train, reverse=True)
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
        font = font_manager.FontProperties(fname=PINGFANG_FONT_FILE_PATH)
    fig, ax = plt.subplots(figsize=(max(8, int(len(classes) / 5)), 5), dpi=120)
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
    ax.set_ylabel('Counts')
    plt.legend()
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "histogram.png")
    fig.savefig(fig_path)
    return {"histogram": "histogram.png"}
