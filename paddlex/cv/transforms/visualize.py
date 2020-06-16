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

import os
import os.path as osp
from .cls_transforms import ClsTransform
from .det_transforms import DetTransform
from .seg_transforms import SegTransform

def visualize(dataset, index=0, steps=3, save_dir='vdl_output'):
    transforms = dataset.transforms
    if not osp.isdir(save_dir):
        if osp.exists(save_dir):
            os.remove(save_dir)
        os.makedirs(save_dir)
    for i, data in enumerate(dataset.iterator()):
        if i == index:
            break
    from visualdl import LogWriter
    vdl_save_dir = osp.join(save_dir, 'image_transforms')
    images_writer = LogWriter(vdl_save_dir)
    data.append(images_writer)
    for s in range(steps):
        if s != 0:
            data.pop()
        data.append(s)
        transforms(*data)
        