#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import paddle
from paddle.io import Dataset
import os
import cv2

from paddlex.ppcls.data import preprocess
from paddlex.ppcls.data.preprocess import transform
from paddlex.ppcls.utils import logger
from .common_dataset import create_operators


class CompCars(Dataset):
    def __init__(self,
                 image_root,
                 cls_label_path,
                 label_root=None,
                 transform_ops=None,
                 bbox_crop=False):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self._label_root = label_root
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self._bbox_crop = bbox_crop
        self._dtype = paddle.get_default_dtype()
        self._load_anno()

    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        if self._bbox_crop:
            assert os.path.exists(self._label_root)
        self.images = []
        self.labels = []
        self.bboxes = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split()
                if not self._bbox_crop:
                    self.images.append(os.path.join(self._img_root, l[0]))
                    self.labels.append(int(l[1]))
                else:
                    label_path = os.path.join(self._label_root,
                                              l[0].split('.')[0] + '.txt')
                    assert os.path.exists(label_path)
                    with open(label_path) as f:
                        bbox = f.readlines()[-1].strip().split()
                    bbox = [int(x) for x in bbox]
                    self.images.append(os.path.join(self._img_root, l[0]))
                    self.labels.append(int(l[1]))
                    self.bboxes.append(bbox)
                    assert os.path.exists(self.images[-1])

    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self._bbox_crop:
            bbox = self.bboxes[idx]
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        if self._transform_ops:
            img = transform(img, self._transform_ops)
        img = img.transpose((2, 0, 1))
        return (img, self.labels[idx])

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))


class VeriWild(Dataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None, ):
        self._img_root = image_root
        self._cls_path = cls_label_path
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self._dtype = paddle.get_default_dtype()
        self._load_anno()

    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []
        self.cameras = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split()
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(np.int64(l[1]))
                self.cameras.append(np.int64(l[2]))
                assert os.path.exists(self.images[-1])

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[idx], self.cameras[idx])
        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))
