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

from .imagenet import ImageNet
from .voc import VOCDetection
from .coco import CocoDetection
from .seg_dataset import SegDataset
from .easydata_cls import EasyDataCls
from .easydata_det import EasyDataDet
from .easydata_seg import EasyDataSeg
from .dataset import generate_minibatch
from .analysis import Seg
from .change_det_dataset import ChangeDetDataset
