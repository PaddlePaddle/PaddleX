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

from __future__ import absolute_import

__version__ = '1.3.6'
gui_mode = True

import os
if 'FLAGS_eager_delete_tensor_gb' not in os.environ:
    os.environ['FLAGS_eager_delete_tensor_gb'] = '0.0'
if 'FLAGS_allocator_strategy' not in os.environ:
    os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
if "CUDA_VISIBLE_DEVICES" in os.environ:
    if os.environ["CUDA_VISIBLE_DEVICES"].count("-1") > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

import paddle
version = paddle.__version__.strip().split('.')
if version[0] == '1':
    if version[1] != '8':
        raise Exception(
            'For running paddlex(v{}), Version of paddlepaddle should be greater than 1.8.3'.
            format(__version__))
    #import paddlehub as hub
    #if hub.__version__.strip().split('.')[0] > '1':
    #    raise Exception("Try to reinstall Paddlehub by 'pip install paddlehub==1.8.2' while paddlepaddle < 2.0")

if hasattr(paddle, 'enable_static'):
    paddle.enable_static()

from .utils.utils import get_environ_info
from . import cv
from . import det
from . import seg
from . import cls
from . import slim
from . import converter
from . import tools
from . import deploy

try:
    import pycocotools
except:
    print(
        "[WARNING] pycocotools is not installed, detection model is not available now."
    )
    print(
        "[WARNING] pycocotools install: https://paddlex.readthedocs.io/zh_CN/develop/install.html#pycocotools"
    )

env_info = get_environ_info()
load_model = cv.models.load_model
datasets = cv.datasets
transforms = cv.transforms

log_level = 2

from . import interpret
