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

import os
import argparse
import base64
import numpy as np


def get_image_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            if single_file.split('.')[-1] in img_end:
                imgs_lists.append(os.path.join(img_file, single_file))
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def get_image_list_from_label_file(image_path, label_file_path):
    imgs_lists = []
    gt_labels = []
    with open(label_file_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            image_name, label = line.strip("\n").split()
            label = int(label)
            imgs_lists.append(os.path.join(image_path, image_name))
            gt_labels.append(int(label))
    return imgs_lists, gt_labels
