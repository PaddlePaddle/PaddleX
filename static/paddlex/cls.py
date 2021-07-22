# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from . import cv

ResNet18 = cv.models.ResNet18
ResNet34 = cv.models.ResNet34
ResNet50 = cv.models.ResNet50
ResNet101 = cv.models.ResNet101
ResNet50_vd = cv.models.ResNet50_vd
ResNet101_vd = cv.models.ResNet101_vd
ResNet50_vd_ssld = cv.models.ResNet50_vd_ssld
ResNet101_vd_ssld = cv.models.ResNet101_vd_ssld
DarkNet53 = cv.models.DarkNet53
MobileNetV1 = cv.models.MobileNetV1
MobileNetV2 = cv.models.MobileNetV2
MobileNetV3_small = cv.models.MobileNetV3_small
MobileNetV3_large = cv.models.MobileNetV3_large
MobileNetV3_small_ssld = cv.models.MobileNetV3_small_ssld
MobileNetV3_large_ssld = cv.models.MobileNetV3_large_ssld
Xception41 = cv.models.Xception41
Xception65 = cv.models.Xception65
DenseNet121 = cv.models.DenseNet121
DenseNet161 = cv.models.DenseNet161
DenseNet201 = cv.models.DenseNet201
ShuffleNetV2 = cv.models.ShuffleNetV2
HRNet_W18 = cv.models.HRNet_W18
AlexNet = cv.models.AlexNet

transforms = cv.transforms.cls_transforms
