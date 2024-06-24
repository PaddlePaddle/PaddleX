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
from random import shuffle
from .....utils.file_interface import custom_open


def split_dataset(dataset_root, train_rate, val_rate):
    """
    将图像数据集按照比例分成训练集、验证集和测试集，并生成对应的.txt文件。
    
    Args:
        dataset_root (str): 数据集根目录路径。
        train_rate (int): 训练集占总数据集的比例（%）。
        val_rate (int): 验证集占总数据集的比例（%）。
    
    Returns:
        str: 数据划分结果信息。
    """
    sum_rate = train_rate + val_rate
    if sum_rate != 100:
        return "训练集、验证集比例之和需要等于100，请修改后重试"
    tags = ["train", "val"]

    valid_path = False
    image_files = []
    for tag in tags:
        split_image_list = os.path.abspath(
            os.path.join(dataset_root, f'{tag}.txt'))
        rename_image_list = os.path.abspath(
            os.path.join(dataset_root, f'{tag}.txt.bak'))
        if os.path.exists(split_image_list):
            with custom_open(split_image_list, 'r') as f:
                lines = f.readlines()
            image_files = image_files + lines
            valid_path = True
            if not os.path.exists(rename_image_list):
                os.rename(split_image_list, rename_image_list)

    if not valid_path:
        return f"数据集目录下保存待划分文件{tags[0]}.txt或{tags[1]}.txt不存在，请检查后重试"

    shuffle(image_files)
    start = 0
    image_num = len(image_files)
    rate_list = [train_rate, val_rate]
    for i, tag in enumerate(tags):

        rate = rate_list[i]
        if rate == 0:
            continue
        if rate > 100 or rate < 0:
            return f"{tag} 数据集的比例应该在0~100之间."

        end = start + round(image_num * rate / 100)
        if sum(rate_list[i + 1:]) == 0:
            end = image_num

        txt_file = os.path.abspath(os.path.join(dataset_root, tag + '.txt'))
        with custom_open(txt_file, 'w') as f:
            m = 0
            for id in range(start, end):
                m += 1
                f.write(image_files[id])
        start = end
    return dataset_root
