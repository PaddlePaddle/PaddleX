# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from . import models
from . import nets
from . import transforms
from . import datasets

cls_transforms = transforms.cls_transforms
det_transforms = transforms.det_transforms
seg_transforms = transforms.seg_transforms

# classification
ResNet50 = models.ResNet50
DarkNet53 = models.DarkNet53
# detection
YOLOv3 = models.YOLOv3
PPYOLO = models.PPYOLO
#EAST = models.EAST
FasterRCNN = models.FasterRCNN
MaskRCNN = models.MaskRCNN
UNet = models.UNet
DeepLabv3p = models.DeepLabv3p
