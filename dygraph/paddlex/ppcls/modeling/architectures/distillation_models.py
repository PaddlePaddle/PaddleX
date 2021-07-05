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
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.nn as nn

from .resnet_vd import ResNet50_vd
from .mobilenet_v3 import MobileNetV3_large_x1_0
from .resnext101_wsl import ResNeXt101_32x16d_wsl

__all__ = [
    'ResNet50_vd_distill_MobileNetV3_large_x1_0',
    'ResNeXt101_32x16d_wsl_distill_ResNet50_vd'
]


class ResNet50_vd_distill_MobileNetV3_large_x1_0(nn.Layer):
    def __init__(self, class_dim=1000, freeze_teacher=True, **args):
        super(ResNet50_vd_distill_MobileNetV3_large_x1_0, self).__init__()

        self.teacher = ResNet50_vd(class_dim=class_dim, **args)
        self.student = MobileNetV3_large_x1_0(class_dim=class_dim, **args)

        if freeze_teacher:
            for param in self.teacher.parameters():
                param.trainable = False

    def forward(self, x):
        teacher_label = self.teacher(x)
        student_label = self.student(x)
        return teacher_label, student_label


class ResNeXt101_32x16d_wsl_distill_ResNet50_vd(nn.Layer):
    def __init__(self, class_dim=1000, freeze_teacher=True, **args):
        super(ResNeXt101_32x16d_wsl_distill_ResNet50_vd, self).__init__()

        self.teacher = ResNeXt101_32x16d_wsl(class_dim=class_dim, **args)
        self.student = ResNet50_vd(class_dim=class_dim, **args)

        if freeze_teacher:
            for param in self.teacher.parameters():
                param.trainable = False

    def forward(self, x):
        teacher_label = self.teacher(x)
        student_label = self.student(x)
        return teacher_label, student_label
