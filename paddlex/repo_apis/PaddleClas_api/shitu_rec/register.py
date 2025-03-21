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
from .config import ShiTuRecConfig
from .model import ShiTuRecModel
from .runner import ShiTuRecRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLECLAS_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "ShiTuRec",
        "model": ShiTuRecModel,
        "runner": ShiTuRecRunner,
        "config": ShiTuRecConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################
register_model_info(
    {
        "model_name": "PP-ShiTuV2_rec",
        "suite": "ShiTuRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-ShiTuV2_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["ShiTuRecDataset"],
        "infer_config": None,
    }
)

register_model_info(
    {
        "model_name": "PP-ShiTuV2_rec_CLIP_vit_base",
        "suite": "ShiTuRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-ShiTuV2_rec_CLIP_vit_base.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["ShiTuRecDataset"],
        "infer_config": None,
    }
)

register_model_info(
    {
        "model_name": "PP-ShiTuV2_rec_CLIP_vit_large",
        "suite": "ShiTuRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-ShiTuV2_rec_CLIP_vit_large.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_dataset_types": ["ShiTuRecDataset"],
        "infer_config": None,
    }
)
