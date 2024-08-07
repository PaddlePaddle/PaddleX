# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import os
import os.path as osp

from ...base.register import register_model_info, register_suite_info
from .model import TextRecModel
from .runner import TextRecRunner
from .config import TextRecConfig

REPO_ROOT_PATH = os.environ.get('PADDLE_PDX_PADDLEOCR_PATH')
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', 'configs'))

register_suite_info({
    'suite_name': 'TextRec',
    'model': TextRecModel,
    'runner': TextRecRunner,
    'config': TextRecConfig,
    'runner_root_path': REPO_ROOT_PATH
})

register_model_info({
    'model_name': 'PP-OCRv4_mobile_rec',
    'suite': 'TextRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-OCRv4_mobile_rec.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export']
})

register_model_info({
    'model_name': 'PP-OCRv4_server_rec',
    'suite': 'TextRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'PP-OCRv4_server_rec.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export']
})

register_model_info({
    'model_name': 'SVTRv2_server_rec',
    'suite': 'TextRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'SVTRv2_server_rec.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})

register_model_info({
    'model_name': 'RepSVTR_mobile_rec',
    'suite': 'TextRec',
    'config_path': osp.join(PDX_CONFIG_DIR, 'RepSVTR_mobile_rec.yaml'),
    'supported_apis': ['train', 'evaluate', 'predict', 'export', 'infer']
})
