#!/usr/bin/env python
# coding: utf-8
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

from .x2imagenet import EasyData2ImageNet
from .x2coco import LabelMe2COCO
from .x2coco import EasyData2COCO
from .x2voc import LabelMe2VOC
from .x2voc import EasyData2VOC
from .x2seg import JingLing2Seg
from .x2seg import LabelMe2Seg
from .x2seg import EasyData2Seg

easydata2imagenet = EasyData2ImageNet().convert
labelme2coco = LabelMe2COCO().convert
easydata2coco = EasyData2COCO().convert
labelme2voc = LabelMe2VOC().convert
easydata2voc = EasyData2VOC().convert
jingling2seg = JingLing2Seg().convert
labelme2seg = LabelMe2Seg().convert
easydata2seg = EasyData2Seg().convert
