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
from ..ts_base.model import TSModel
from .config import TSClassifyConfig
from .runner import TSCLSRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLETS_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "TSClassify",
        "model": TSModel,
        "runner": TSCLSRunner,
        "config": TSClassifyConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################

# timesnet
TimesNetCLS_CFG_PATH = osp.join(PDX_CONFIG_DIR, "TimesNet_cls.yaml")
register_model_info(
    {
        "model_name": "TimesNet_cls",
        "suite": "TSClassify",
        "config_path": TimesNetCLS_CFG_PATH,
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_n1cx", "xpu", "npu", "mlu"],
            "dy2st": False,
            "amp": [],
        },
        "supported_evaluate_opts": {
            "device": ["cpu", "gpu_n1cx", "xpu", "npu", "mlu"],
            "amp": [],
        },
        "supported_predict_opts": {"device": ["cpu", "gpu", "xpu", "npu", "mlu"]},
        "supported_infer_opts": {"device": ["cpu", "gpu", "xpu", "npu", "mlu"]},
    }
)
