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
from .config import TSAnomalyConfig
from .runner import TSADRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLETS_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "TSAnomaly",
        "model": TSModel,
        "runner": TSADRunner,
        "config": TSAnomalyConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

################ Models Using Universal Config ################

# timesnet
TimesNetAD_CFG_PATH = osp.join(PDX_CONFIG_DIR, "TimesNet_ad.yaml")
register_model_info(
    {
        "model_name": "TimesNet_ad",
        "suite": "TSAnomaly",
        "config_path": TimesNetAD_CFG_PATH,
        "auto_compression_config_path": TimesNetAD_CFG_PATH,
        "supported_apis": ["train", "evaluate", "predict", "export"],
        "supported_train_opts": {
            "device": ["cpu", "gpu_n1cx"],
            "dy2st": False,
            "amp": [],
        },
        "supported_evaluate_opts": {"device": ["cpu", "gpu_n1cx"], "amp": []},
        "supported_predict_opts": {"device": ["cpu", "gpu"]},
        "supported_infer_opts": {"device": ["cpu", "gpu"]},
        "supported_compression_opts": {"device": ["cpu", "gpu_n1cx"]},
    }
)

# autoencoder_anomaly
AE_CFG_PATH = osp.join(PDX_CONFIG_DIR, "AutoEncoder_ad.yaml")
register_model_info(
    {
        "model_name": "AutoEncoder_ad",
        "suite": "TSAnomaly",
        "config_path": AE_CFG_PATH,
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

# dlinear_anomaly
DL_CFG_PATH = osp.join(PDX_CONFIG_DIR, "DLinear_ad.yaml")
register_model_info(
    {
        "model_name": "DLinear_ad",
        "suite": "TSAnomaly",
        "config_path": DL_CFG_PATH,
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

# patch_anomaly
PATCHTST_CFG_PATH = osp.join(PDX_CONFIG_DIR, "PatchTST_ad.yaml")
register_model_info(
    {
        "model_name": "PatchTST_ad",
        "suite": "TSAnomaly",
        "config_path": PATCHTST_CFG_PATH,
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

# ns_anomaly
NS_CFG_PATH = osp.join(PDX_CONFIG_DIR, "Nonstationary_ad.yaml")
register_model_info(
    {
        "model_name": "Nonstationary_ad",
        "suite": "TSAnomaly",
        "config_path": NS_CFG_PATH,
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
