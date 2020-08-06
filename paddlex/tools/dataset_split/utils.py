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
import os.path as osp
from PIL import Image
import numpy as np
import json


class MyEncoder(json.JSONEncoder):
    # 调整json文件存储形式
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def list_files(dirname):
    """ 列出目录下所有文件（包括所属的一级子目录下文件）

    Args:
        dirname: 目录路径
    """

    def filter_file(f):
        if f.startswith('.'):
            return True
        return False

    all_files = list()
    dirs = list()
    for f in os.listdir(dirname):
        if filter_file(f):
            continue
        if osp.isdir(osp.join(dirname, f)):
            dirs.append(f)
        else:
            all_files.append(f)
    for d in dirs:
        for f in os.listdir(osp.join(dirname, d)):
            if filter_file(f):
                continue
            if osp.isdir(osp.join(dirname, d, f)):
                continue
            all_files.append(osp.join(d, f))
    return all_files


def is_pic(filename):
    """ 判断文件是否为图片格式

    Args:
        filename: 文件路径
    """
    suffixes = {'JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png'}
    suffix = filename.strip().split('.')[-1]
    if suffix not in suffixes:
        return False
    return True


def replace_ext(filename, new_ext):
    """ 替换文件后缀

    Args:
        filename: 文件路径
        new_ext: 需要替换的新的后缀
    """
    items = filename.split(".")
    items[-1] = new_ext
    new_filename = ".".join(items)
    return new_filename


def read_seg_ann(pngfile):
    """ 解析语义分割的标注png图片

    Args:
        pngfile: 包含标注信息的png图片路径
    """
    grt = np.asarray(Image.open(pngfile))
    labels = list(np.unique(grt))
    if 255 in labels:
        labels.remove(255)
    return labels
