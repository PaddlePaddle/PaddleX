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
from ...image_classification.modeling._config_pplcnetv4 import PPLCNetV4Config


class PPOCRV6SmallRecConfig(PretrainedConfig):
    model_type = "pp_ocrv6_small_rec"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", None)
        if isinstance(backbone_config, PPLCNetV4Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = PPLCNetV4Config(**backbone_config)
        else:
            self.backbone_config = PPLCNetV4Config()

        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 120)
        self.mlp_ratio = kwargs.get("mlp_ratio", 2.0)
        self.depth = kwargs.get("depth", 2)
        self.head_out_channels = kwargs.get("head_out_channels", 18714)
        self.conv_kernel_size = kwargs.get("conv_kernel_size", [1, 7])
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-6)
