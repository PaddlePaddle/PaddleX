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

from .classifier import BaseClassifier
from .classifier import ResNet18
from .classifier import ResNet34
from .classifier import ResNet50
from .classifier import ResNet101
from .classifier import ResNet50_vd
from .classifier import ResNet101_vd
from .classifier import ResNet50_vd_ssld
from .classifier import ResNet101_vd_ssld
from .classifier import DarkNet53
from .classifier import MobileNetV1
from .classifier import MobileNetV2
from .classifier import MobileNetV3_small
from .classifier import MobileNetV3_large
from .classifier import MobileNetV3_small_ssld
from .classifier import MobileNetV3_large_ssld
from .classifier import Xception41
from .classifier import Xception65
from .classifier import DenseNet121
from .classifier import DenseNet161
from .classifier import DenseNet201
from .classifier import ShuffleNetV2
from .classifier import HRNet_W18
from .base import BaseAPI
from .yolo_v3 import YOLOv3
from .faster_rcnn import FasterRCNN
from .mask_rcnn import MaskRCNN
from .unet import UNet
from .deeplabv3p import DeepLabv3p
from .hrnet import HRNet
from .load_model import load_model
from .slim import prune
