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
import shutil
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from .....utils.errors import ConvertFailedError


def check_src_dataset(root_dir):
    """ check src dataset format validity """

    err_msg_prefix = f"数据格式转换失败！当前仅支持后续为'.xlsx/.xls'格式的数据转换。"

    for dst_anno, src_anno in [("train.xlsx", "train.xls"),
                               ("val.xlsx", "val.xls"),
                               ("test.xlsx", "test.xls")]:
        src_anno_path = os.path.join(root_dir, src_anno)
        dst_anno_path = os.path.join(root_dir, dst_anno)
        if not os.path.exists(src_anno_path) and not os.path.exists(
                dst_anno_path):
            if 'train' in dst_anno:
                raise ConvertFailedError(
                    message=f"{err_msg_prefix}保证{src_anno_path}或{dst_anno_path}文件存在。"
                )
            continue


def convert_excel_dataset(input_dir, output_dir):
    """
    将excel标注的数据集转换为PaddleX需要的格式
    
    Args:
        input_dir (str): 输入的目录，包含多个json格式的Labelme标注文件
        output_dir (str): 输出的目录，转换后的图片和标注文件都会存储在此目录下
    
    Returns:
        str: 返回一个字符串表示转换的结果，“转换成功”表示转换没有问题。
    
    Raises:
        该函数目前没有特定的异常抛出。
    
    """
    # prepare dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read excel file
    for dst_anno, src_anno in [("train.xlsx", "train.xls"),
                               ("val.xlsx", "val.xls"),
                               ("test.xlsx", "test.xls")]:
        src_anno_path = os.path.join(input_dir, src_anno)
        dst_anno_path = os.path.join(input_dir, dst_anno)

        if os.path.exists(src_anno_path):
            excel_file = pd.read_excel(src_anno_path)
            output_csv_dir = os.path.join(output_dir,
                                          src_anno.replace(".xlsx", ".csv"))
            excel_file.to_csv(output_csv_dir, index=False)
        if os.path.exists(dst_anno_path):
            excel_file = pd.read_excel(dst_anno_path)
            output_csv_dir = os.path.join(output_dir,
                                          dst_anno.replace(".xlsx", ".csv"))
            excel_file.to_csv(output_csv_dir, index=False)


def convert(input_dir, output_dir):
    """ convert dataset to coco format """
    # check format validity
    check_src_dataset(input_dir)
    convert_excel_dataset(input_dir, output_dir)
    return output_dir
