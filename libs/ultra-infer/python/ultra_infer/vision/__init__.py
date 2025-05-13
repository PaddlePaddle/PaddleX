# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from . import (
    classification,
    detection,
    evaluation,
    facealign,
    facedet,
    faceid,
    generation,
    headpose,
    keypointdetection,
    matting,
    ocr,
    perception,
    segmentation,
    sr,
    tracking,
)
from .utils import fd_result_to_json
from .visualize import *
from .. import C


def enable_flycv():
    return C.vision.enable_flycv()


def disable_flycv():
    return C.vision.disable_flycv()
