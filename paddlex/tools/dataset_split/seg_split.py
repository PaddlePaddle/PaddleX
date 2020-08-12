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
from .utils import list_files, is_pic, replace_ext, read_seg_ann
import paddlex.utils.logging as logging


def split_seg_dataset(dataset_dir, val_percent, test_percent, save_dir):
    if not osp.exists(osp.join(dataset_dir, "JPEGImages")):
        logging.error("\'JPEGImages\' is not found in {}!".format(dataset_dir))
    if not osp.exists(osp.join(dataset_dir, "Annotations")):
        logging.error("\'Annotations\' is not found in {}!".format(
            dataset_dir))

    all_image_files = list_files(osp.join(dataset_dir, "JPEGImages"))

    image_anno_list = list()
    label_list = list()
    for image_file in all_image_files:
        if not is_pic(image_file):
            continue
        anno_name = replace_ext(image_file, "png")
        if osp.exists(osp.join(dataset_dir, "Annotations", anno_name)):
            image_anno_list.append([image_file, anno_name])
        else:
            anno_name = replace_ext(image_file, "PNG")
            if osp.exists(osp.join(dataset_dir, "Annotations", anno_name)):
                image_anno_list.append([image_file, anno_name])
            else:
                logging.error("The annotation file {} doesn't exist!".format(
                    anno_name))

    if not osp.exists(osp.join(dataset_dir, "labels.txt")):
        for image_anno in image_anno_list:
            labels = read_seg_ann(
                osp.join(dataset_dir, "Annotations", anno_name))
            for i in labels:
                if i not in label_list:
                    label_list.append(i)
        # 如果类标签的最大值大于类别数，添加对应缺失的标签
        if len(label_list) != max(label_list) + 1:
            label_list = [i for i in range(max(label_list) + 1)]

    random.shuffle(image_anno_list)
    image_num = len(image_anno_list)
    val_num = int(image_num * val_percent)
    test_num = int(image_num * test_percent)
    train_num = image_num - val_num - test_num

    train_image_anno_list = image_anno_list[:train_num]
    val_image_anno_list = image_anno_list[train_num:train_num + val_num]
    test_image_anno_list = image_anno_list[train_num + val_num:]

    with open(
            osp.join(save_dir, 'train_list.txt'), mode='w',
            encoding='utf-8') as f:
        for x in train_image_anno_list:
            file = osp.join("JPEGImages", x[0])
            label = osp.join("Annotations", x[1])
            f.write('{} {}\n'.format(file, label))
    with open(
            osp.join(save_dir, 'val_list.txt'), mode='w',
            encoding='utf-8') as f:
        for x in val_image_anno_list:
            file = osp.join("JPEGImages", x[0])
            label = osp.join("Annotations", x[1])
            f.write('{} {}\n'.format(file, label))
    if len(test_image_anno_list):
        with open(
                osp.join(save_dir, 'test_list.txt'), mode='w',
                encoding='utf-8') as f:
            for x in test_image_anno_list:
                file = osp.join("JPEGImages", x[0])
                label = osp.join("Annotations", x[1])
                f.write('{} {}\n'.format(file, label))
    if len(label_list):
        with open(
                osp.join(save_dir, 'labels.txt'), mode='w',
                encoding='utf-8') as f:
            for l in sorted(label_list):
                f.write('{}\n'.format(l))

    return train_num, val_num, test_num
