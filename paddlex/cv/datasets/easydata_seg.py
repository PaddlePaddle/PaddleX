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
import cv2
import numpy as np
import paddlex.utils.logging as logging
from .dataset import Dataset
from .dataset import get_encoding
from .dataset import is_pic

class EasyDataSeg(Dataset):
    """读取EasyDataSeg语义分割任务数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路）。
        label_list (str): 描述数据集包含的类别信息文件路径。
        transforms (list): 数据集中每个样本的预处理/增强算子。
        num_workers (int): 数据集中样本在预处理过程中的线程或进程数。默认为4。
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
        super(EasyDataSeg, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        self.file_list = list()
        self.labels = list()
        self._epoch = 0

        from pycocotools.mask import decode
        cname2cid = {}
        label_id = 0
        with open(label_list, encoding=get_encoding(label_list)) as fr:
            for line in fr.readlines():
                cname2cid[line.strip()] = label_id
                label_id += 1
                self.labels.append(line.strip())
                
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
                im = cv2.imread(img_file)
                im_w = im.shape[1]
                im_h = im.shape[0]
                objs = json_info['labels']
                lable_npy = np.zeros([im_h, im_w]).astype('uint8')
                for i, obj in enumerate(objs):
                    cname = obj['name']
                    cid = cname2cid[cname]
                    mask_dict = {}
                    mask_dict['size'] = [im_h, im_w]
                    mask_dict['counts'] = obj['mask'].encode()
                    mask = decode(mask_dict)
                    mask *= cid
                    conflict_index = np.where(((lable_npy > 0) & (mask == cid)) == True)
                    mask[conflict_index] = 0
                    lable_npy += mask
                self.file_list.append([img_file, lable_npy])
        self.num_samples = len(self.file_list)
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))

    def iterator(self):
        self._epoch += 1
        self._pos = 0
        files = copy.deepcopy(self.file_list)
        if self.shuffle:
            random.shuffle(files)
        files = files[:self.num_samples]
        self.num_samples = len(files)
        for f in files:
            lable_npy = f[1]
            sample = [f[0], None, lable_npy]
            yield sample
