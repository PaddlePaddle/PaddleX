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

from .contrib.resnet import ResNet
from .contrib.yolov5cls import YOLOv5Cls
from .ppcls import *
from .ppshitu import (
    PPShiTuV2Detector,
    PPShiTuV2Recognizer,
    PPShiTuV2RecognizerPostprocessor,
    PPShiTuV2RecognizerPreprocessor,
)

PPLCNet = PaddleClasModel
PPLCNetv2 = PaddleClasModel
EfficientNet = PaddleClasModel
GhostNet = PaddleClasModel
MobileNetv1 = PaddleClasModel
MobileNetv2 = PaddleClasModel
MobileNetv3 = PaddleClasModel
ShuffleNetv2 = PaddleClasModel
SqueezeNet = PaddleClasModel
Inceptionv3 = PaddleClasModel
PPHGNet = PaddleClasModel
ResNet50vd = PaddleClasModel
SwinTransformer = PaddleClasModel
