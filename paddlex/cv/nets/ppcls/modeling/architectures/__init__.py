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

from .alexnet import AlexNet
from .darknet import DarkNet53
from .mobilenet_v1 import *
from .mobilenet_v2 import *
from .mobilenet_v3 import *
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .resnet_vd import ResNet18_vd, ResNet34_vd, ResNet50_vd, ResNet101_vd, ResNet152_vd, ResNet200_vd
from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, DenseNet264
from .hrnet import *
from .xception import Xception41, Xception65, Xception71
from .shufflenet_v2 import *
