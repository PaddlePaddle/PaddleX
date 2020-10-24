# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import six
import sys
import random
import numpy as np
import paddlex.utils.logging as logging
import paddlex as pst
from .voc import VOCDetection
from .dataset import is_pic


class CocoDetection(VOCDetection):
    """读取MSCOCO格式的检测数据集，并对样本进行相应的处理，该格式的数据集同样可以应用到实例分割模型的训练中。

    Args:
        data_dir (str): 数据集所在的目录路径。
        ann_file (str): 数据集的标注文件，为一个独立的json格式文件。
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
        from pycocotools.coco import COCO

        try:
            import shapely.ops
            from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        except:
            six.reraise(*sys.exc_info())

        super(VOCDetection, self).__init__(
            transforms=transforms,
            num_workers=num_workers,
            buffer_size=buffer_size,
            parallel_method=parallel_method,
            shuffle=shuffle)
        self.file_list = list()
        self.labels = list()
        self._epoch = 0

        coco = COCO(ann_file)
        self.coco_gt = coco
        img_ids = coco.getImgIds()
        cat_ids = coco.getCatIds()
        catid2clsid = dict({catid: i + 1 for i, catid in enumerate(cat_ids)})
        cname2cid = dict({
            coco.loadCats(catid)[0]['name']: clsid
            for catid, clsid in catid2clsid.items()
        })
        for label, cid in sorted(cname2cid.items(), key=lambda d: d[1]):
            self.labels.append(label)
        logging.info("Starting to read file list from dataset...")
        for img_id in img_ids:
            img_anno = coco.loadImgs(img_id)[0]
            im_fname = osp.join(data_dir, img_anno['file_name'])
            if not is_pic(im_fname):
                continue
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])
            ins_anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
            instances = coco.loadAnns(ins_anno_ids)

            bboxes = []
            for inst in instances:
                x, y, box_w, box_h = inst['bbox']
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(im_w - 1, x1 + max(0, box_w - 1))
                y2 = min(im_h - 1, y1 + max(0, box_h - 1))
                if inst['area'] > 0 and x2 >= x1 and y2 >= y1:
                    inst['clean_bbox'] = [x1, y1, x2, y2]
                    bboxes.append(inst)
                else:
                    logging.warning(
                        "Found an invalid bbox in annotations: im_id: {}, area: {} x1: {}, y1: {}, x2: {}, y2: {}."
                        .format(img_id, float(inst['area']), x1, y1, x2, y2))
            num_bbox = len(bboxes)
            gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
            gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_score = np.ones((num_bbox, 1), dtype=np.float32)
            is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
            difficult = np.zeros((num_bbox, 1), dtype=np.int32)
            gt_poly = [None] * num_bbox

            for i, box in enumerate(bboxes):
                catid = box['category_id']
                gt_class[i][0] = catid2clsid[catid]
                gt_bbox[i, :] = box['clean_bbox']
                is_crowd[i][0] = box['iscrowd']
                if 'segmentation' in box:
                    gt_poly[i] = box['segmentation']

            im_info = {
                'im_id': np.array([img_id]).astype('int32'),
                'image_shape': np.array([im_h, im_w]).astype('int32'),
            }
            label_info = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'gt_poly': gt_poly,
                'difficult': difficult
            }

            if None in gt_poly:
                del label_info['gt_poly']

            coco_rec = (im_info, label_info)
            self.file_list.append([im_fname, coco_rec])
        if not len(self.file_list) > 0:
            raise Exception('not found any coco record in %s' % (ann_file))
        logging.info("{} samples in file {}".format(
            len(self.file_list), ann_file))
        self.num_samples = len(self.file_list)
