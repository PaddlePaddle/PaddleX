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
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
import paddlex.utils.logging as logging
from .dataset import Dataset
from .dataset import is_pic
from .dataset import get_encoding


class VOCDetection(Dataset):
    """读取PascalVOC格式的检测数据集，并对样本进行相应的处理。

    Args:
        data_dir (str): 数据集所在的目录路径。
        file_list (str): 描述数据集图片文件和对应标注文件的文件路径（文本内每行路径为相对data_dir的相对路）。
        label_list (str): 描述数据集包含的类别信息文件路径。
        transforms (paddlex.det.transforms): 数据集中每个样本的预处理/增强算子。
        num_workers (int|str): 数据集中样本在预处理过程中的线程或进程数。默认为'auto'。当设为'auto'时，根据
            系统的实际CPU核数设置`num_workers`: 如果CPU核数的一半大于8，则`num_workers`为8，否则为CPU核数的
            一半。
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
        from pycocotools.coco import COCO
        super(VOCDetection, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        self.file_list = list()
        self.labels = list()
        self._epoch = 0

        annotations = {}
        annotations['images'] = []
        annotations['categories'] = []
        annotations['annotations'] = []

        cname2cid = OrderedDict()
        label_id = 1
        with open(label_list, 'r', encoding=get_encoding(label_list)) as fr:
            for line in fr.readlines():
                cname2cid[line.strip()] = label_id
                label_id += 1
                self.labels.append(line.strip())
        logging.info("Starting to read file list from dataset...")
        for k, v in cname2cid.items():
            annotations['categories'].append({
                'supercategory': 'component',
                'id': v,
                'name': k
            })
        ct = 0
        ann_ct = 0
        with open(file_list, 'r', encoding=get_encoding(file_list)) as fr:
            while True:
                line = fr.readline()
                if not line:
                    break
                img_file, xml_file = [osp.join(data_dir, x) \
                        for x in line.strip().split()[:2]]
                if not is_pic(img_file):
                    continue
                if not osp.isfile(xml_file):
                    continue
                if not osp.exists(img_file):
                    raise IOError('The image file {} is not exist!'.format(
                        img_file))
                tree = ET.parse(xml_file)
                if tree.find('id') is None:
                    im_id = np.array([ct])
                else:
                    ct = int(tree.find('id').text)
                    im_id = np.array([int(tree.find('id').text)])

                objs = tree.findall('object')
                im_w = float(tree.find('size').find('width').text)
                im_h = float(tree.find('size').find('height').text)
                gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
                gt_class = np.zeros((len(objs), 1), dtype=np.int32)
                gt_score = np.ones((len(objs), 1), dtype=np.float32)
                is_crowd = np.zeros((len(objs), 1), dtype=np.int32)
                difficult = np.zeros((len(objs), 1), dtype=np.int32)
                for i, obj in enumerate(objs):
                    cname = obj.find('name').text.strip()
                    gt_class[i][0] = cname2cid[cname]
                    _difficult = int(obj.find('difficult').text)
                    x1 = float(obj.find('bndbox').find('xmin').text)
                    y1 = float(obj.find('bndbox').find('ymin').text)
                    x2 = float(obj.find('bndbox').find('xmax').text)
                    y2 = float(obj.find('bndbox').find('ymax').text)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    if im_w > 0.5 and im_h > 0.5:
                        x2 = min(im_w - 1, x2)
                        y2 = min(im_h - 1, y2)
                    gt_bbox[i] = [x1, y1, x2, y2]
                    is_crowd[i][0] = 0
                    difficult[i][0] = _difficult
                    annotations['annotations'].append({
                        'iscrowd': 0,
                        'image_id': int(im_id[0]),
                        'bbox': [x1, y1, x2 - x1 + 1, y2 - y1 + 1],
                        'area': float((x2 - x1 + 1) * (y2 - y1 + 1)),
                        'category_id': cname2cid[cname],
                        'id': ann_ct,
                        'difficult': _difficult
                    })
                    ann_ct += 1

                im_info = {
                    'im_id': im_id,
                    'image_shape': np.array([im_h, im_w]).astype('int32'),
                }
                label_info = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_score': gt_score,
                    'difficult': difficult
                }
                voc_rec = (im_info, label_info)
                if len(objs) != 0:
                    self.file_list.append([img_file, voc_rec])
                    ct += 1
                    annotations['images'].append({
                        'height': im_h,
                        'width': im_w,
                        'id': int(im_id[0]),
                        'file_name': osp.split(img_file)[1]
                    })

        if not len(self.file_list) > 0:
            raise Exception('not found any voc record in %s' % (file_list))
        logging.info("{} samples in file {}".format(
            len(self.file_list), file_list))
        self.num_samples = len(self.file_list)
        self.coco_gt = COCO()
        self.coco_gt.dataset = annotations
        self.coco_gt.createIndex()

    def iterator(self):
        self._epoch += 1
        self._pos = 0
        files = copy.deepcopy(self.file_list)
        if self.shuffle:
            random.shuffle(files)
        files = files[:self.num_samples]
        self.num_samples = len(files)
        for f in files:
            records = f[1]
            im_info = copy.deepcopy(records[0])
            label_info = copy.deepcopy(records[1])
            im_info['epoch'] = self._epoch
            if self.num_samples > 1:
                mix_idx = random.randint(1, self.num_samples - 1)
                mix_pos = (mix_idx + self._pos) % self.num_samples
            else:
                mix_pos = 0
            im_info['mixup'] = [
                files[mix_pos][0], copy.deepcopy(files[mix_pos][1][0]),
                copy.deepcopy(files[mix_pos][1][1])
            ]
            self._pos += 1
            sample = [f[0], im_info, label_info]
            yield sample
