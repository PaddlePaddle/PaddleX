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
from .cv.models.utils.visualize import visualize_segmentation
from paddlex.cv.transforms import seg_transforms

transforms = seg_transforms

UNet = cv.models.UNet
DeepLabv3p = cv.models.DeepLabV3P
HRNet = cv.models.HRNet
FastSCNN = cv.models.FastSCNN

visualize = visualize_segmentation
