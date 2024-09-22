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
from pathlib import Path

from ...base.register import register_model_info, register_suite_info
from .model import TableRecModel
from .runner import TableRecRunner
from .config import TableRecConfig

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLEOCR_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))
HPI_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "utils" / "hpi_configs"

register_suite_info(
    {
        "suite_name": "TableRec",
        "model": TableRecModel,
        "runner": TableRecRunner,
        "config": TableRecConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

register_model_info(
    {
        "model_name": "SLANet",
        "suite": "TableRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "SLANet.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "hpi_config_path": HPI_CONFIG_DIR / "SLANet.yaml",
    }
)
