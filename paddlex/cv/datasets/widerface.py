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
import copy
import os.path as osp
import random
import cv2
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
import paddlex.utils.logging as logging
from .voc import VOCDetection
from .dataset import is_pic
from .dataset import get_encoding

class WIDERFACEDetection(VOCDetection):
    """读取WIDER Face格式的检测数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        ann_file (str): 数据集的标注文件，为一个独立的txt格式文件。
        transforms (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据
            系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的一半。
        buffer_size (int): 数据集中样本在预处理过程中队列的缓存长度，以样本数为单位。默认为100。
        parallel_method (str): 数据集中样本在预处理过程中并行处理的方式，支持'thread'
            线程和'process'进程两种方式。默认为'process'（Windows和Mac下会强制使用thread，该参数无效）。
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
    """
    
    def __init__(self,
                 data_dir,
                 ann_file,
                 transforms=None,
                 num_workers='auto',
                 buffer_size=100,
                 parallel_method='process',
                 shuffle=False):
        super(VOCDetection, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        
        self.file_list = list()
        self.labels = list()
        self._epoch = 0
        self.labels.append('face')
        valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
        
        from pycocotools.coco import COCO
        annotations = {}
        annotations['images'] = []
        annotations['categories'] = []
        annotations['annotations'] = []
        annotations['categories'].append({
                'supercategory': 'component',
                'id': 1,
                'name': 'face'
            })
        logging.info("Starting to read file list from dataset...")
        im_ct = 0
        ann_ct = 0
        is_discard = False
        with open(ann_file, 'r', encoding=get_encoding(ann_file)) as fr:
            lines_txt = fr.readlines()
            for line in lines_txt:
                line = line.strip('\n\t\r')
                if any(suffix in line for suffix in valid_suffix):
                    img_file = osp.join(data_dir, line)
                    if not is_pic(img_file):
                        is_discard = False
                        continue
                    else:
                        is_discard = True
                    im = cv2.imread(img_file)
                    im_w = im.shape[1]
                    im_h = im.shape[0]
                    im_info = {
                        'im_id': np.array([im_ct]),
                        'image_shape': np.array([im_h, im_w]).astype('int32'),
                    }
                    bbox_id = 0
                    annotations['images'].append({
                        'height':
                        im_h,
                        'width':
                        im_w,
                        'id':
                        im_ct,
                        'file_name':
                        osp.split(img_file)[1]
                    })
                elif ' ' not in line:
                    if not is_discard:
                        continue
                    bbox_ct = int(line)
                    if bbox_ct == 0:
                        is_discard = False
                        continue
                    gt_bbox = np.zeros((bbox_ct, 4), dtype=np.float32)
                    gt_class = np.ones((bbox_ct, 1), dtype=np.int32)
                    difficult = np.zeros((bbox_ct, 1), dtype=np.int32)
                else:
                    if not is_discard:
                        continue
                    split_str = line.split(' ')
                    xmin = float(split_str[0])
                    ymin = float(split_str[1])
                    w = float(split_str[2])
                    h = float(split_str[3])
                    # Filter out wrong labels
                    if w < 0 or h < 0:
                        logging.warning('Illegal box with w: {}, h: {} in '
                                    'img: {}, and it will be ignored'.format(
                                        w, h, img_file))
                        gt_class[bbox_id, 0] = 0
                        bbox_id += 1
                        continue
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = xmin + w
                    ymax = ymin + h
                    gt_bbox[bbox_id] = [xmin, ymin, xmax, ymax]
                    bbox_id += 1
                    annotations['annotations'].append({
                        'iscrowd': 0,
                        'image_id': im_ct,
                        'bbox': [xmin, ymin, w, h],
                        'area': float(w * h),
                        'category_id': 1,
                        'id': ann_ct,
                        'difficult': 0
                    })
                    ann_ct += 1
                    if bbox_id == bbox_ct:
                        label_info = {
                            'gt_class': gt_class,
                            'gt_bbox': gt_bbox,
                            'difficult': difficult
                        }
                        voc_rec = (im_info, label_info)
                        self.file_list.append([img_file, voc_rec])
                        im_ct += 1
                        
        self.num_samples = len(self.file_list)
        self.coco_gt = COCO()
        self.coco_gt.dataset = annotations
        self.coco_gt.createIndex()
                        
                    