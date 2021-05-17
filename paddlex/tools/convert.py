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

from .dataset_conversion import *

easydata2imagenet = EasyData2ImageNet().convert
jingling2imagenet = JingLing2ImageNet().convert
jingling2seg = JingLing2Seg().convert
labelme2seg = LabelMe2Seg().convert
easydata2seg = EasyData2Seg().convert
labelme2voc = LabelMe2VOC().convert
easydata2voc = EasyData2VOC().convert
labelme2coco = LabelMe2COCO().convert
easydata2coco = EasyData2COCO().convert
jingling2coco = JingLing2COCO().convert


def dataset_conversion(source, to, pics, anns, save_dir):
    if source.lower() == 'easydata' and to.lower() == 'imagenet':
        easydata2imagenet(pics, anns, save_dir)
    elif source.lower() == 'jingling' and to.lower() == 'imagenet':
        jingling2imagenet(pics, anns, save_dir)
    elif source.lower() == 'jingling' and to.lower() == 'seg':
        jingling2seg(pics, anns, save_dir)
    elif source.lower() == 'labelme' and to.lower() == 'seg':
        labelme2seg(pics, anns, save_dir)
    elif source.lower() == 'easydata' and to.lower() == 'seg':
        easydata2seg(pics, anns, save_dir)
    elif source.lower() == 'labelme' and to.lower() == 'voc':
        labelme2voc(pics, anns, save_dir)
    elif source.lower() == 'easydata' and to.lower() == 'voc':
        easydata2voc(pics, anns, save_dir)
    elif source.lower() == 'labelme' and to.lower() == 'coco':
        labelme2coco(pics, anns, save_dir)
    elif source.lower() == 'easydata' and to.lower() == 'coco':
        easydata2coco(pics, anns, save_dir)
    elif source.lower() == 'jingling' and to.lower() == 'coco':
        jingling2coco(pics, anns, save_dir)
    else:
        raise Exception("Converting from {} to {} is not supported.".format(
            source, to))
