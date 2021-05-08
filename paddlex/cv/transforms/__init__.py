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

from .operators import *
from .batch_operators import BatchRandomResize, BatchRandomResizeByShort, _BatchPadding
import paddlex.cv.transforms as T


def arrange_transforms(model_type, transforms, mode='train'):
    # 给transforms添加arrange操作
    if model_type == 'segmenter':
        if mode == 'eval':
            transforms.apply_im_only = True
        else:
            transforms.apply_im_only = False
        arrange_transform = ArrangeSegmenter(mode)
    elif model_type == 'classifier':
        arrange_transform = ArrangeClassifier(mode)
    elif model_type == 'detector':
        arrange_transform = ArrangeDetector(mode)
    else:
        raise Exception("Unrecognized model type: {}".format(model_type))
    transforms.arrange_outputs = arrange_transform


def build_transforms(transforms_info):
    transforms = list()
    for op_info in transforms_info:
        op_name = list(op_info.keys())[0]
        op_attr = op_info[op_name]
        if not hasattr(T, op_name):
            raise Exception("There's no transform named '{}'".format(op_name))
        transforms.append(getattr(T, op_name)(**op_attr))
    eval_transforms = T.Compose(transforms)
    return eval_transforms
