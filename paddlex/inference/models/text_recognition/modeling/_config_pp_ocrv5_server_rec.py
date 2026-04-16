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
from ...image_classification.modeling._config_hgnetv2 import HGNetV2Config

DEFAULT_REC_BACKBONE_CONFIG = {
    "model_type": "hgnet_v2",
    "num_channels": 3,
    "embedding_size": 64,
    "hidden_sizes": [256, 512, 1024, 2048],
    "hidden_act": "relu",
    "num_labels": 0,
    "stem_channels": [3, 32, 48],
    "stem_strides": [2, 1, 1, 1, 1],
    "stage_in_channels": [48, 128, 512, 1024],
    "stage_mid_channels": [48, 96, 192, 384],
    "stage_out_channels": [128, 512, 1024, 2048],
    "stage_num_blocks": [1, 1, 3, 1],
    "stage_downsample": [True, True, True, True],
    "stage_downsample_strides": [[2, 1], [1, 2], [2, 1], [2, 1]],
    "stage_light_block": [False, False, True, True],
    "stage_kernel_size": [3, 3, 5, 5],
    "stage_numb_of_layers": [6, 6, 6, 6],
    "use_learnable_affine_block": False,
}


class PPOCRV5ServerRecConfig(PretrainedConfig):
    model_type = "pp_ocrv5_server_rec"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", DEFAULT_REC_BACKBONE_CONFIG)
        if isinstance(backbone_config, HGNetV2Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = HGNetV2Config(**backbone_config)
        else:
            self.backbone_config = HGNetV2Config(**DEFAULT_REC_BACKBONE_CONFIG)

        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.hidden_size = kwargs.get("hidden_size", 120)
        self.mlp_ratio = kwargs.get("mlp_ratio", 2.0)
        self.depth = kwargs.get("depth", 2)
        self.head_out_channels = kwargs.get("head_out_channels", 18385)
        self.conv_kernel_size = kwargs.get("conv_kernel_size", [1, 3])
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-6)
