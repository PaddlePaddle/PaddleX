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

import os
import os.path as osp

from ...base.register import register_model_info, register_suite_info
from .config import VideoClsConfig
from .model import VideoClsModel
from .runner import VideoClsRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLEVIDEO_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "VideoCls",
        "model": VideoClsModel,
        "runner": VideoClsRunner,
        "config": VideoClsConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################
register_model_info(
    {
        "model_name": "PP-TSM-R50_8frames_uniform",
        "suite": "VideoCls",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-TSM-R50_8frames_uniform.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)

register_model_info(
    {
        "model_name": "PP-TSMv2-LCNetV2_8frames_uniform",
        "suite": "VideoCls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "PP-TSMv2-LCNetV2_8frames_uniform.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)


register_model_info(
    {
        "model_name": "PP-TSMv2-LCNetV2_16frames_uniform",
        "suite": "VideoCls",
        "config_path": osp.join(
            PDX_CONFIG_DIR, "PP-TSMv2-LCNetV2_16frames_uniform.yaml"
        ),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
        "infer_config": "deploy/configs/inference_cls.yaml",
    }
)
