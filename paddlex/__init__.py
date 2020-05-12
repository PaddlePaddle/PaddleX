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

from __future__ import absolute_import
from .utils.utils import get_environ_info
from . import cv
from . import det
from . import seg
from . import cls
from . import slim
from . import tools

try:
    import pycocotools
except:
    print("[WARNING] pycocotools is not installed, detection model is not available now.")
    print("[WARNING] pycocotools install: https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/install.md")

import paddlehub as hub
if hub.version.hub_version < '1.6.2':
    raise Exception("[ERROR] paddlehub >= 1.6.2 is required")


env_info = get_environ_info()
load_model = cv.models.load_model
datasets = cv.datasets

log_level = 2
__version__ = '0.1.6.github'
