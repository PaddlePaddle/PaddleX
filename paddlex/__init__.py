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

__version__ = '2.0.0rc3'

from paddlex.utils.env import get_environ_info, init_parallel_env
init_parallel_env()

from . import cv
from . import seg
from . import cls
from . import det
from . import tools

from .cv.models.utils.visualize import visualize_detection as visualize_det
from .cv.models.utils.visualize import visualize_segmentation as visualize_seg

env_info = get_environ_info()
datasets = cv.datasets
transforms = cv.transforms

log_level = 2

load_model = cv.models.load_model

visualize_det = visualize_det
visualize_seg = visualize_seg
