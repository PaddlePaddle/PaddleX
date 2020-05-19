# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
import os.path as osp
import random
import copy
import json
import paddlex.utils.logging as logging
from .imagenet import ImageNet
from .dataset import is_pic
from .dataset import get_encoding


class EasyDataCls(ImageNet):
    """读取EasyDataCls格式的分类数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路）。
        label_list (str): 描述数据集包含的类别信息文件路径。
        transforms (paddlex.cls.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据
            系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核
            数的一半。
        buffer_size (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。
        parallel_method (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'
            线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
    """
    
    def __init__(self,
                 data_dir,
                 file_list,
                 label_list,
                 transforms=None,
                 num_workers='auto',
                 buffer_size=100,
                 parallel_method='process',
                 shuffle=False):
        super(ImageNet, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        self.file_list = list()
        self.labels = list()
        self._epoch = 0
        
        with open(label_list, encoding=get_encoding(label_list)) as f:
            for line in f:
                item = line.strip()
                self.labels.append(item)
        logging.info("Starting to read file list from dataset...")
        with open(file_list, encoding=get_encoding(file_list)) as f:
            for line in f:
                img_file, json_file = [osp.join(data_dir, x) \
                        for x in line.strip().split()[:2]]
                if not is_pic(img_file):
                    continue
                if not osp.isfile(json_file):
                    continue
                if not osp.exists(img_file):
                    raise IOError(
                        'The image file {} is not exist!'.format(img_file))
                with open(json_file, mode='r', \
                          encoding=get_encoding(json_file)) as j:
                    json_info = json.load(j)
                label = json_info['labels'][0]['name']
                self.file_list.append([img_file, self.labels.index(label)])
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))
    