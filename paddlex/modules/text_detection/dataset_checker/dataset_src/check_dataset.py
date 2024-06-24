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

from .....utils.errors import DatasetFileNotFoundError


def check(dataset_dir, output, sample_num=10):
    """ check dataset """
    dataset_dir = osp.abspath(dataset_dir)

    if not osp.exists(dataset_dir) or not osp.isdir(dataset_dir):
        raise DatasetFileNotFoundError(file_path=dataset_dir)

    sample_cnts = dict()
    sample_paths = defaultdict(list)
    delim = '\t'
    valid_num_parts = 2

    tags = ['train', 'val']
    for _, tag in enumerate(tags):
        file_list = osp.join(dataset_dir, f'{tag}.txt')
        if not osp.exists(file_list):
            if tag in ('train', 'val'):
                # train and val file lists must exist
                raise DatasetFileNotFoundError(
                    file_path=file_list,
                    solution=f"Ensure that both `train.txt` and `val.txt` exist in \
{dataset_dir}")
            else:
                continue
        else:
            with open(file_list, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                sample_cnts[tag] = len(all_lines)
                for idx, line in enumerate(all_lines):
                    substr = line.strip("\n").split(delim)
                    if len(line.strip("\n")) < 1:
                        continue
                    assert len(substr) == valid_num_parts or len(
                            line.strip("\n")) <= 1, \
                                f"Error in {line}, \
                                The number of delimiter-separated items in each row in {file_list} \
                                should be {valid_num_parts} (current delimiter is '{delim}')."

                    file_name = substr[0]
                    label = substr[1]
                    img_path = osp.join(dataset_dir, file_name)
                    if len(sample_paths[tag]) < sample_num:
                        sample_paths[tag].append(
                            os.path.relpath(img_path, output))
                    if not osp.exists(img_path):
                        raise DatasetFileNotFoundError(file_path=img_path)

                    # check det label
                    label = json.loads(label)
                    for item in label:
                        assert "points" in item and "transcription" in item, \
                            f"line {idx} is not in the correct format."
                        box = np.array(item['points'])
                        assert box.shape == (4, 2), \
                            f"{box} in line {idx} is not in the correct format."

                        txt = item['transcription']
                        assert isinstance(txt, str), \
                            f"{txt} in line {idx} is not in the correct format."

    attrs = {}
    attrs['train_samples'] = sample_cnts['train']
    attrs['train_sample_paths'] = sample_paths['train'][:sample_num]

    attrs['val_samples'] = sample_cnts['val']
    attrs['val_sample_paths'] = sample_paths['val'][:sample_num]
    return attrs
