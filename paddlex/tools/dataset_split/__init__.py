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

from .imagenet_split import split_imagenet_dataset
from .seg_split import split_seg_dataset
from .voc_split import split_voc_dataset
from .coco_split import split_coco_dataset

__all__ = [
    'split_imagenet_dataset', 'split_seg_dataset', 'split_voc_dataset',
    'split_coco_dataset'
]
