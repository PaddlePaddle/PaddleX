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

# Defaults mirror the released PP-OCRv6_medium_det config.json so bare-init
# builds a self-consistent, forward-able model — following the same
# `_config_pp_ocrv5_server.py` convention. (HF's PPOCRV6MediumDetConfig
# defaults `intraclass_block_config / scale_factor_list / kernel_list` to
# `None`; released config.json files always override them.)
DEFAULT_BACKBONE_CONFIG = {
    "model_type": "pp_lcnet_v4",
    "stem_channels": [3, 64, 128],
    "stem_type": "large",
    "block_configs": [
        [[3, 128, 128, 1, True], [3, 128, 128, 1, False]],
        [
            [3, 128, 256, 2, False],
            [3, 256, 256, 1, True],
            [3, 256, 256, 1, False],
        ],
        [
            [3, 256, 512, 2, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        [
            [3, 512, 896, 2, False],
            [3, 896, 896, 1, True],
            [3, 896, 896, 1, False],
        ],
    ],
}

DEFAULT_INTRACLASS_BLOCK_CONFIG = {
    "reduce_channel": [1, 1, 0],
    "return_channel": [1, 1, 0],
    "vertical_long_to_small_conv_longratio": [[7, 1], [1, 1], [3, 0]],
    "vertical_long_to_small_conv_midratio": [[5, 1], [1, 1], [2, 0]],
    "vertical_long_to_small_conv_shortratio": [[3, 1], [1, 1], [1, 0]],
    "horizontal_small_to_long_conv_longratio": [[1, 7], [1, 1], [0, 3]],
    "horizontal_small_to_long_conv_midratio": [[1, 5], [1, 1], [0, 2]],
    "horizontal_small_to_long_conv_shortratio": [[1, 3], [1, 1], [0, 1]],
    "symmetric_conv_long_longratio": [[7, 7], [1, 1], [3, 3]],
    "symmetric_conv_long_midratio": [[5, 5], [1, 1], [2, 2]],
    "symmetric_conv_long_shortratio": [[3, 3], [1, 1], [1, 1]],
}


class PPOCRV6MediumDetConfig(PretrainedConfig):
    model_type = "pp_ocrv6_medium_det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", DEFAULT_BACKBONE_CONFIG)
        if isinstance(backbone_config, PPLCNetV4Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = PPLCNetV4Config(**backbone_config)
        else:
            self.backbone_config = PPLCNetV4Config(**DEFAULT_BACKBONE_CONFIG)

        self.interpolate_mode = kwargs.get("interpolate_mode", "nearest")
        self.neck_out_channels = kwargs.get("neck_out_channels", 256)
        self.reduce_factor = kwargs.get("reduce_factor", 2)
        self.intraclass_block_number = kwargs.get("intraclass_block_number", 4)
        self.intraclass_block_config = kwargs.get(
            "intraclass_block_config", DEFAULT_INTRACLASS_BLOCK_CONFIG
        )
        self.scale_factor = kwargs.get("scale_factor", 2)
        self.scale_factor_list = kwargs.get("scale_factor_list", [1, 2, 4, 8])
        self.kernel_list = kwargs.get("kernel_list", [3, 2, 2])
        self.hidden_act = kwargs.get("hidden_act", "relu")
