# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import os.path as osp
import random
import re
import numpy as np
from collections import OrderedDict
import xml.etree.ElementTree as ET
from paddle.io import Dataset
from paddlex.utils import logging, get_num_workers, get_encoding, path_normalization, is_pic
from paddlex.cv.transforms import Decode, MixupImage
from paddlex.tools import YOLOAnchorCluster


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
        shuffle (bool): 是否需要对数据集中样本打乱顺序。默认为False。
        allow_empty (bool): 是否加载负样本。默认为False。
        empty_ratio (float): 用于指定负样本占总样本数的比例。如果小于0或大于等于1，则保留全部的负样本。默认为1。
    """

    def __init__(self,
                 data_dir,
                 file_list,
                 label_list,
                 transforms=None,
                 num_workers='auto',
                 shuffle=False,
                 allow_empty=False,
                 empty_ratio=1.):
        # matplotlib.use() must be called *before* pylab, matplotlib.pyplot,
        # or matplotlib.backends is imported for the first time
        # pycocotools import matplotlib
        import matplotlib
        matplotlib.use('Agg')
        from pycocotools.coco import COCO
        super(VOCDetection, self).__init__()
        self.data_dir = data_dir
        self.data_fields = None
        self.transforms = copy.deepcopy(transforms)
        self.num_max_boxes = 50

        self.use_mix = False
        if self.transforms is not None:
            for op in self.transforms.transforms:
                if isinstance(op, MixupImage):
                    self.mixup_op = copy.deepcopy(op)
                    self.use_mix = True
                    self.num_max_boxes *= 2
                    break

        self.batch_transforms = None
        self.num_workers = get_num_workers(num_workers)
        self.shuffle = shuffle
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio
        self.file_list = list()
        neg_file_list = list()
        self.labels = list()

        annotations = dict()
        annotations['images'] = list()
        annotations['categories'] = list()
        annotations['annotations'] = list()

        cname2cid = OrderedDict()
        label_id = 0
        with open(label_list, 'r', encoding=get_encoding(label_list)) as f:
            for line in f.readlines():
                cname2cid[line.strip()] = label_id
                label_id += 1
                self.labels.append(line.strip())
        logging.info("Starting to read file list from dataset...")
        for k, v in cname2cid.items():
            annotations['categories'].append({
                'supercategory': 'component',
                'id': v + 1,
                'name': k
            })
        ct = 0
        ann_ct = 0
        with open(file_list, 'r', encoding=get_encoding(file_list)) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if len(line.strip().split()) > 2:
                    raise Exception("A space is defined as the separator, "
                                    "but it exists in image or label name {}."
                                    .format(line))
                img_file, xml_file = [
                    osp.join(data_dir, x) for x in line.strip().split()[:2]
                ]
                img_file = path_normalization(img_file)
                xml_file = path_normalization(xml_file)
                if not is_pic(img_file):
                    continue
                if not osp.isfile(xml_file):
                    continue
                if not osp.exists(img_file):
                    logging.warning('The image file {} does not exist!'.format(
                        img_file))
                    continue
                if not osp.exists(xml_file):
                    logging.warning('The annotation file {} does not exist!'.
                                    format(xml_file))
                    continue
                tree = ET.parse(xml_file)
                if tree.find('id') is None:
                    im_id = np.asarray([ct])
                else:
                    ct = int(tree.find('id').text)
                    im_id = np.asarray([int(tree.find('id').text)])
                pattern = re.compile('<size>', re.IGNORECASE)
                size_tag = pattern.findall(
                    str(ET.tostringlist(tree.getroot())))
                if len(size_tag) > 0:
                    size_tag = size_tag[0][1:-1]
                    size_element = tree.find(size_tag)
                    pattern = re.compile('<width>', re.IGNORECASE)
                    width_tag = pattern.findall(
                        str(ET.tostringlist(size_element)))[0][1:-1]
                    im_w = float(size_element.find(width_tag).text)
                    pattern = re.compile('<height>', re.IGNORECASE)
                    height_tag = pattern.findall(
                        str(ET.tostringlist(size_element)))[0][1:-1]
                    im_h = float(size_element.find(height_tag).text)
                else:
                    im_w = 0
                    im_h = 0

                pattern = re.compile('<object>', re.IGNORECASE)
                obj_match = pattern.findall(
                    str(ET.tostringlist(tree.getroot())))
                if len(obj_match) > 0:
                    obj_tag = obj_match[0][1:-1]
                    objs = tree.findall(obj_tag)
                else:
                    objs = list()

                num_bbox, i = len(objs), 0
                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_score = np.zeros((num_bbox, 1), dtype=np.float32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                difficult = np.zeros((num_bbox, 1), dtype=np.int32)
                for obj in objs:
                    pattern = re.compile('<name>', re.IGNORECASE)
                    name_tag = pattern.findall(str(ET.tostringlist(obj)))[0][
                        1:-1]
                    cname = obj.find(name_tag).text.strip()
                    pattern = re.compile('<difficult>', re.IGNORECASE)
                    diff_tag = pattern.findall(str(ET.tostringlist(obj)))
                    if len(diff_tag) == 0:
                        _difficult = 0
                    else:
                        diff_tag = diff_tag[0][1:-1]
                        try:
                            _difficult = int(obj.find(diff_tag).text)
                        except Exception:
                            _difficult = 0
                    pattern = re.compile('<bndbox>', re.IGNORECASE)
                    box_tag = pattern.findall(str(ET.tostringlist(obj)))
                    if len(box_tag) == 0:
                        logging.warning(
                            "There's no field '<bndbox>' in one of object, "
                            "so this object will be ignored. xml file: {}".
                            format(xml_file))
                        continue
                    box_tag = box_tag[0][1:-1]
                    box_element = obj.find(box_tag)
                    pattern = re.compile('<xmin>', re.IGNORECASE)
                    xmin_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    x1 = float(box_element.find(xmin_tag).text)
                    pattern = re.compile('<ymin>', re.IGNORECASE)
                    ymin_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    y1 = float(box_element.find(ymin_tag).text)
                    pattern = re.compile('<xmax>', re.IGNORECASE)
                    xmax_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    x2 = float(box_element.find(xmax_tag).text)
                    pattern = re.compile('<ymax>', re.IGNORECASE)
                    ymax_tag = pattern.findall(
                        str(ET.tostringlist(box_element)))[0][1:-1]
                    y2 = float(box_element.find(ymax_tag).text)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    if im_w > 0.5 and im_h > 0.5:
                        x2 = min(im_w - 1, x2)
                        y2 = min(im_h - 1, y2)

                    if not (x2 >= x1 and y2 >= y1):
                        logging.warning(
                            "Bounding box for object {} does not satisfy x1 <= x2 and y1 <= y2, "
                            "so this object is skipped. xml file: {}".format(i, xml_file))
                        continue

                    gt_bbox[i, :] = [x1, y1, x2, y2]
                    gt_class[i, 0] = cname2cid[cname]
                    gt_score[i, 0] = 1.
                    is_crowd[i, 0] = 0
                    difficult[i, 0] = _difficult
                    i += 1
                    annotations['annotations'].append({
                        'iscrowd': 0,
                        'image_id': int(im_id[0]),
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'area': float((x2 - x1) * (y2 - y1)),
                        'category_id': cname2cid[cname] + 1,
                        'id': ann_ct,
                        'difficult': _difficult
                    })
                    ann_ct += 1

                gt_bbox = gt_bbox[:i, :]
                gt_class = gt_class[:i, :]
                gt_score = gt_score[:i, :]
                is_crowd = is_crowd[:i, :]
                difficult = difficult[:i, :]

                im_info = {
                    'im_id': im_id,
                    'image_shape': np.array(
                        [im_h, im_w], dtype=np.int32)
                }
                label_info = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    'gt_bbox': gt_bbox,
                    'gt_score': gt_score,
                    'difficult': difficult
                }

                if gt_bbox.size > 0:
                    self.file_list.append({
                        'image': img_file,
                        **
                        im_info,
                        **
                        label_info
                    })
                    annotations['images'].append({
                        'height': im_h,
                        'width': im_w,
                        'id': int(im_id[0]),
                        'file_name': osp.split(img_file)[1]
                    })
                else:
                    neg_file_list.append({
                        'image': img_file,
                        **
                        im_info,
                        **
                        label_info
                    })
                ct += 1

                if self.use_mix:
                    self.num_max_boxes = max(self.num_max_boxes, 2 * len(objs))
                else:
                    self.num_max_boxes = max(self.num_max_boxes, len(objs))

        if not ct:
            logging.error(
                "No voc record found in %s' % (file_list)", exit=True)
        self.pos_num = len(self.file_list)
        if self.allow_empty and neg_file_list:
            self.file_list += self._sample_empty(neg_file_list)
        logging.info(
            "{} samples in file {}, including {} positive samples and {} negative samples.".
            format(
                len(self.file_list), file_list, self.pos_num,
                len(self.file_list) - self.pos_num))
        self.num_samples = len(self.file_list)
        self.coco_gt = COCO()
        self.coco_gt.dataset = annotations
        self.coco_gt.createIndex()

        self._epoch = 0

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.file_list[idx])
        if self.data_fields is not None:
            sample = {k: sample[k] for k in self.data_fields}
        if self.use_mix and (self.mixup_op.mixup_epoch == -1 or
                             self._epoch < self.mixup_op.mixup_epoch):
            if self.num_samples > 1:
                mix_idx = random.randint(1, self.num_samples - 1)
                mix_pos = (mix_idx + idx) % self.num_samples
            else:
                mix_pos = 0
            sample_mix = copy.deepcopy(self.file_list[mix_pos])
            if self.data_fields is not None:
                sample_mix = {k: sample_mix[k] for k in self.data_fields}
            sample = self.mixup_op(sample=[
                Decode(to_rgb=False)(sample), Decode(to_rgb=False)(sample_mix)
            ])
        sample = self.transforms(sample)
        return sample

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch_id):
        self._epoch = epoch_id

    def cluster_yolo_anchor(self,
                            num_anchors,
                            image_size,
                            cache=True,
                            cache_path=None,
                            iters=300,
                            gen_iters=1000,
                            thresh=.25):
        """
        Cluster YOLO anchors.

        Reference:
            https://github.com/ultralytics/yolov5/blob/master/utils/autoanchor.py

        Args:
            num_anchors (int): number of clusters
            image_size (list or int): [h, w], being an int means image height and image width are the same.
            cache (bool): whether using cache
            cache_path (str or None, optional): cache directory path. If None, use `data_dir` of dataset.
            iters (int, optional): iters of kmeans algorithm
            gen_iters (int, optional): iters of genetic algorithm
            threshold (float, optional): anchor scale threshold
            verbose (bool, optional): whether print results
        """
        if cache_path is None:
            cache_path = self.data_dir
        cluster = YOLOAnchorCluster(
            num_anchors=num_anchors,
            dataset=self,
            image_size=image_size,
            cache=cache,
            cache_path=cache_path,
            iters=iters,
            gen_iters=gen_iters,
            thresh=thresh)
        anchors = cluster()
        return anchors

    def add_negative_samples(self, image_dir, empty_ratio=1):
        """将背景图片加入训练

        Args:
            image_dir (str)：背景图片所在的文件夹目录。
            empty_ratio (float or None): 用于指定负样本占总样本数的比例。如果为None，保留数据集初始化是设置的`empty_ratio`值，
                否则更新原有`empty_ratio`值。如果小于0或大于等于1，则保留全部的负样本。默认为1。

        """
        import cv2
        if not osp.isdir(image_dir):
            raise Exception("{} is not a valid image directory.".format(
                image_dir))
        if empty_ratio is not None:
            self.empty_ratio = empty_ratio
        image_list = os.listdir(image_dir)
        max_img_id = max(
            len(self.file_list) - 1, max(self.coco_gt.getImgIds()))
        neg_file_list = list()
        for image in image_list:
            if not is_pic(image):
                continue
            gt_bbox = np.zeros((0, 4), dtype=np.float32)
            gt_class = np.zeros((0, 1), dtype=np.int32)
            gt_score = np.zeros((0, 1), dtype=np.float32)
            is_crowd = np.zeros((0, 1), dtype=np.int32)
            difficult = np.zeros((0, 1), dtype=np.int32)

            max_img_id += 1
            im_fname = osp.join(image_dir, image)
            img_data = cv2.imread(im_fname, cv2.IMREAD_UNCHANGED)
            im_h, im_w, im_c = img_data.shape

            im_info = {
                'im_id': np.asarray([max_img_id]),
                'image_shape': np.array(
                    [im_h, im_w], dtype=np.int32)
            }
            label_info = {
                'is_crowd': is_crowd,
                'gt_class': gt_class,
                'gt_bbox': gt_bbox,
                'gt_score': gt_score,
                'difficult': difficult
            }
            if 'gt_poly' in self.file_list[0]:
                label_info['gt_poly'] = []

            neg_file_list.append({
                'image': im_fname,
                **
                im_info,
                **
                label_info
            })
        if neg_file_list:
            self.allow_empty = True
            self.file_list += self._sample_empty(neg_file_list)
        logging.info(
            "{} negative samples added. Dataset contains {} positive samples and {} negative samples.".
            format(
                len(self.file_list) - self.num_samples, self.pos_num,
                len(self.file_list) - self.pos_num))
        self.num_samples = len(self.file_list)

    def _sample_empty(self, neg_file_list):
        if 0. <= self.empty_ratio < 1.:
            import random
            total_num = len(self.file_list)
            neg_num = total_num - self.pos_num
            sample_num = min((total_num * self.empty_ratio - neg_num) //
                             (1 - self.empty_ratio), len(neg_file_list))
            return random.sample(neg_file_list, sample_num)
        else:
            return neg_file_list
