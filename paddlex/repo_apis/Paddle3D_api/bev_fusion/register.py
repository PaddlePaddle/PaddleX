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
from .config import BEVFusionConfig
from .model import BEVFusionModel
from .runner import BEVFusionRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLE3D_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "BEVFusion",
        "model": BEVFusionModel,
        "runner": BEVFusionRunner,
        "config": BEVFusionConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

register_model_info(
    {
        "model_name": "BEVFusion",
        "suite": "BEVFusion",
        "config_path": osp.join(PDX_CONFIG_DIR, "BEVFusion.yaml"),
        "auto_compression_config_path": osp.join(PDX_CONFIG_DIR, "None"),
        "supported_apis": ["train", "evaluate", "export", "infer"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_nxcx"],
            "dy2st": False,
            "amp": ["O1", "O2"],
        },
        "supported_evaluate_opts": {"device": ["cpu", "gpu_nxcx"]},
        "supported_infer_opts": {"device": ["cpu", "gpu"]},
        "supported_dataset_types": [],
        # Additional info
        "infer_dir": "deploy/bevfusion",
    }
)
