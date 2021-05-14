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

from . import cv

ResNet18 = cv.models.ResNet18
ResNet34 = cv.models.ResNet34
ResNet50 = cv.models.ResNet50
ResNet101 = cv.models.ResNet101
ResNet152 = cv.models.ResNet152

ResNet18_vd = cv.models.ResNet18_vd
ResNet34_vd = cv.models.ResNet34_vd
ResNet50_vd = cv.models.ResNet50_vd
ResNet50_vd_ssld = cv.models.ResNet50_vd_ssld
ResNet101_vd = cv.models.ResNet101_vd
ResNet101_vd_ssld = cv.models.ResNet101_vd_ssld
ResNet152_vd = cv.models.ResNet152_vd
ResNet200_vd = cv.models.ResNet200_vd

MobileNetV1 = cv.models.MobileNetV1
MobileNetV2 = cv.models.MobileNetV2
MobileNetV3_small = cv.models.MobileNetV3_small
MobileNetV3_large = cv.models.MobileNetV3_large

AlexNet = cv.models.AlexNet

DarkNet53 = cv.models.DarkNet53

DenseNet121 = cv.models.DenseNet121
DenseNet161 = cv.models.DenseNet161
DenseNet169 = cv.models.DenseNet169
DenseNet201 = cv.models.DenseNet201
DenseNet264 = cv.models.DenseNet264

HRNet_W18_C = cv.models.HRNet_W18_C
HRNet_W30_C = cv.models.HRNet_W30_C
HRNet_W32_C = cv.models.HRNet_W32_C
HRNet_W40_C = cv.models.HRNet_W40_C
HRNet_W44_C = cv.models.HRNet_W44_C
HRNet_W48_C = cv.models.HRNet_W48_C
HRNet_W64_C = cv.models.HRNet_W64_C

Xception41 = cv.models.Xception41
Xception65 = cv.models.Xception65
Xception71 = cv.models.Xception71

ShuffleNetV2 = cv.models.ShuffleNetV2
ShuffleNetV2_swish = cv.models.ShuffleNetV2_swish
