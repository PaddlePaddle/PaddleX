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
from collections import defaultdict
from .....utils.file_interface import custom_open


def simple_analyse(dataset_path):
    """
    Analyse the dataset samples by return image path and label path

    Args:
        dataset_path (str): dataset path

    Returns:
        tuple: tuple of sample number, image path and label path for train, val and text subdataset.
    
    """
    tags = ['train', 'val', 'test']
    sample_cnts = defaultdict(int)
    img_paths = defaultdict(list)
    res = [None] * 6

    for tag in tags:
        file_list = os.path.join(dataset_path, f'{tag}.txt')
        if not os.path.exists(file_list):
            if tag in ('train', 'val'):
                res.insert(0, "数据集不符合规范，请先通过数据校准")
                return res
            else:
                continue
        else:
            with custom_open(file_list, 'r') as f:
                all_lines = f.readlines()

            # Each line corresponds to a sample
            sample_cnts[tag] = len(all_lines)
            # img_paths[tag] = images_dict[tag]

    return f"训练数据样本数: {sample_cnts[tags[0]]}\t评估数据样本数: {sample_cnts[tags[1]]}"


def deep_analyse(dataset_path, output=None):
    """class analysis for dataset"""
    return simple_analyse(dataset_path)
