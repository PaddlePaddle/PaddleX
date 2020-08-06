# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
import random
from .utils import list_files, is_pic
import paddlex.utils.logging as logging


def split_imagenet_dataset(dataset_dir, val_percent, test_percent, save_dir):
    all_files = list_files(dataset_dir)
    label_list = list()
    train_image_anno_list = list()
    val_image_anno_list = list()
    test_image_anno_list = list()
    for file in all_files:
        if not is_pic(file):
            continue
        label, image_name = osp.split(file)
        if label not in label_list:
            label_list.append(label)
    label_list = sorted(label_list)

    for i in range(len(label_list)):
        image_list = list_files(osp.join(dataset_dir, label_list[i]))
        image_anno_list = list()
        for img in image_list:
            image_anno_list.append([osp.join(label_list[i], img), i])
        random.shuffle(image_anno_list)
        image_num = len(image_anno_list)
        val_num = int(image_num * val_percent)
        test_num = int(image_num * test_percent)
        train_num = image_num - val_num - test_num

        train_image_anno_list += image_anno_list[:train_num]
        val_image_anno_list += image_anno_list[train_num:train_num + val_num]
        test_image_anno_list += image_anno_list[train_num + val_num:]

    with open(
            osp.join(save_dir, 'train_list.txt'), mode='w',
            encoding='utf-8') as f:
        for x in train_image_anno_list:
            file, label = x
            f.write('{} {}\n'.format(file, label))
    with open(
            osp.join(save_dir, 'val_list.txt'), mode='w',
            encoding='utf-8') as f:
        for x in val_image_anno_list:
            file, label = x
            f.write('{} {}\n'.format(file, label))
    if len(test_image_anno_list):
        with open(
                osp.join(save_dir, 'test_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in test_image_anno_list:
                file, label = x
                f.write('{} {}\n'.format(file, label))
    with open(
            osp.join(save_dir, 'labels.txt'), mode='w', encoding='utf-8') as f:
        for l in sorted(label_list):
            f.write('{}\n'.format(l))

    return len(train_image_anno_list), len(val_image_anno_list), len(
        test_image_anno_list)
