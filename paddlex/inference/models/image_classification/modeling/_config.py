# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ...common.transformers.transformers import PretrainedConfig

DEFAULT_CONFIG = {
    "model_name": "PP-LCNet_x1_0_doc_ori",
    "scale": 1.0,
    "class_num": 4,
    "stride_list": [2, 2, 2, 2, 2],
    "dropout_prob": 0.2,
    "class_expand": 1280,
    "use_last_conv": True,
    "act": "hardswish",
    "reduction": 4,
    "lr_mult_list": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "net_config": {
        "blocks2": [[3, 16, 32, 1, False]],
        "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
        "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
        "blocks5": [
            [3, 128, 256, 2, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
        ],
        "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
    },
}


class PPLCNetConfig(PretrainedConfig):
    model_type = "cls"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs.get("model_name", DEFAULT_CONFIG["model_name"])
        self.scale = kwargs.get("scale", DEFAULT_CONFIG["scale"])
        self.class_num = kwargs.get("class_num", DEFAULT_CONFIG["class_num"])
        self.stride_list = kwargs.get("stride_list", DEFAULT_CONFIG["stride_list"])
        self.reduction = kwargs.get("reduction", DEFAULT_CONFIG["reduction"])
        self.dropout_prob = kwargs.get("dropout_prob", DEFAULT_CONFIG["dropout_prob"])
        self.class_expand = kwargs.get("class_expand", DEFAULT_CONFIG["class_expand"])
        self.use_last_conv = kwargs.get(
            "use_last_conv", DEFAULT_CONFIG["use_last_conv"]
        )
        self.act = kwargs.get("act", DEFAULT_CONFIG["act"])
        self.lr_mult_list = kwargs.get("lr_mult_list", DEFAULT_CONFIG["lr_mult_list"])
        self.net_config = kwargs.get("net_config", DEFAULT_CONFIG["net_config"])
