# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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


_DEFAULT_BACKBONE_CONFIG = {
    "model_type": "pp_lcnet",
    "scale": 1,
    "out_features": ["stage2", "stage3", "stage4", "stage5"],
    "out_indices": [2, 3, 4, 5],
    "divisor": 16,
}


class SLANetConfig(PretrainedConfig):
    """Configuration for SLANet / SLANet_plus table structure recognition.

    Mirrors ``transformers.models.slanet.SLANetConfig`` so the Paddle model
    loads directly from HF safetensors.
    """

    model_type = "slanet"

    def __init__(self, **kwargs):
        backbone_config = kwargs.pop("backbone_config", None)
        if backbone_config is None:
            backbone_config = dict(_DEFAULT_BACKBONE_CONFIG)
        elif isinstance(backbone_config, dict):
            merged = dict(_DEFAULT_BACKBONE_CONFIG)
            merged.update(backbone_config)
            backbone_config = merged
        self.backbone_config = backbone_config

        self.post_conv_out_channels = kwargs.get("post_conv_out_channels", 96)
        self.out_channels = kwargs.get("out_channels", 50)
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.max_text_length = kwargs.get("max_text_length", 500)

        self.hidden_act = kwargs.get("hidden_act", "hardswish")
        self.csp_kernel_size = kwargs.get("csp_kernel_size", 5)
        self.csp_num_blocks = kwargs.get("csp_num_blocks", 1)

        # PaddleX postprocess expects an 8-coord loc output slot; HF SLANet
        # doesn't predict locations so we fill it with zeros at forward time.
        self.loc_reg_num = kwargs.get("loc_reg_num", 8)
        self.tensor_parallel_degree = 1

        super().__init__(**kwargs)
