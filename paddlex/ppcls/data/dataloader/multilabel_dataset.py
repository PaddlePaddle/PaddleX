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
import os
import cv2

from paddlex.ppcls.data.preprocess import transform
from paddlex.ppcls.utils import logger

from .common_dataset import CommonDataset


class MultiLabelDataset(CommonDataset):
    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split("\t")
                self.images.append(os.path.join(self._img_root, l[0]))

                labels = l[1].split(',')
                labels = [np.int64(i) for i in labels]

                self.labels.append(labels)
                assert os.path.exists(self.images[-1])

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            label = np.array(self.labels[idx]).astype("float32")
            return (img, label)

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
