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

import sys
from . import cv
from .cv.models.utils.visualize import visualize_segmentation

message = 'Your running script needs PaddleX<2.0.0, please refer to {} to solve this issue.'.format(
    'https://github.com/PaddlePaddle/PaddleX/tree/release/2.0-rc/tutorials/train#%E7%89%88%E6%9C%AC%E5%8D%87%E7%BA%A7'
)


def __getattr__(attr):
    if attr == 'transforms':

        print("\033[1;31;40m{}\033[0m".format(message).encode("utf-8")
              .decode("latin1"))
        sys.exit(-1)


visualize = visualize_segmentation

UNet = cv.models.UNet
DeepLabV3P = cv.models.DeepLabV3P
FastSCNN = cv.models.FastSCNN
HRNet = cv.models.HRNet
BiSeNetV2 = cv.models.BiSeNetV2
