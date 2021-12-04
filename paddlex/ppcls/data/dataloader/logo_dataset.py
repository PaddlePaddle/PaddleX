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

import io
import tarfile
import numpy as np
from PIL import Image  #all use default backend

import paddle
from paddle.io import Dataset
import pickle
import os
import cv2
import random

from .common_dataset import CommonDataset


class LogoDataset(CommonDataset):
    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split("\t")
                if l[0] == 'image_id':
                    continue
                self.images.append(os.path.join(self._img_root, l[3]))
                self.labels.append(np.int64(l[1]) - 1)
                assert os.path.exists(self.images[-1])
