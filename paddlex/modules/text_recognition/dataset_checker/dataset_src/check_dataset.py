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
import os.path as osp
from collections import defaultdict

from PIL import Image
import json
import numpy as np

from .....utils.errors import DatasetFileNotFoundError, CheckFailedError


def check(dataset_dir,
          output,
          dataset_type="MSTextRecDataset",
          mode='fast',
          sample_num=10):
    """ check dataset """
    # dataset_dir = osp.abspath(dataset_dir)
    if dataset_type == 'SimpleDataSet' or 'MSTextRecDataset':
        # Custom dataset
        if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
            raise DatasetFileNotFoundError(file_path=dataset_dir)

        tags = ['train', 'val']
        delim = '\t'
        valid_num_parts = 2
        max_recorded_sample_cnts = 50
        sample_cnts = dict()
        sample_paths = defaultdict(list)

        dict_file = osp.join(dataset_dir, 'dict.txt')
        if not osp.exists(dict_file):
            raise DatasetFileNotFoundError(
                file_path=dict_file,
                solution=f"Ensure that `dict.txt` exist in {dataset_dir}")
        for tag in tags:
            file_list = osp.join(dataset_dir, f'{tag}.txt')
            if not osp.exists(file_list):
                if tag in ('train', 'val'):
                    # train and val file lists must exist
                    raise DatasetFileNotFoundError(
                        file_path=file_list,
                        solution=f"Ensure that both `train.txt` and `val.txt` exist in {dataset_dir}"
                    )
                else:
                    # tag == 'test'
                    continue
            else:
                with open(file_list, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    sample_cnts[tag] = len(all_lines)
                    for line in all_lines:
                        substr = line.strip("\n").split(delim)
                        if len(line.strip("\n")) < 1:
                            continue
                        if len(substr) != valid_num_parts and len(
                                line.strip("\n")) > 1:
                            raise CheckFailedError(
                                f"Error in {line}, The number of delimiter-separated items in each row "\
                                "in {file_list} should be {valid_num_parts} (current delimiter is '{delim}')."
                            )
                        file_name = substr[0]
                        img_path = osp.join(dataset_dir, file_name)
                        if len(sample_paths[tag]) < max_recorded_sample_cnts:
                            sample_paths[tag].append(
                                os.path.relpath(img_path, output))

                        if not os.path.exists(img_path):
                            raise DatasetFileNotFoundError(file_path=img_path)

        meta = {}
        meta['train_samples'] = sample_cnts['train']
        meta['train_sample_paths'] = sample_paths['train'][:sample_num]

        meta['val_samples'] = sample_cnts['val']
        meta['val_sample_paths'] = sample_paths['val'][:sample_num]

        # meta['dict_file'] = dict_file

        return meta
