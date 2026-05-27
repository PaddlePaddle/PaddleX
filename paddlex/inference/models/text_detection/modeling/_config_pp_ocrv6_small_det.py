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

# Defaults mirror the released PP-OCRv6_small_det config.json, the canonical
# variant for this class. This follows the same convention as
# `_config_pp_ocrv5_mobile.py` / `_config_pp_ocrv5_server.py`: bare-init
# builds a self-consistent, forward-able model that matches the released
# checkpoint architecture. (HF's `PPOCRV6SmallDetConfig` defaults
# `layer_list_out_channels=(12, 18, 42, 360)` are placeholder values that
# don't compose with the default `PPLCNetV4Config()` backbone — released
# config.json files always override them and so do these defaults.)
DEFAULT_BACKBONE_CONFIG = {
    "model_type": "pp_lcnet_v4",
    "stem_channels": [3, 24, 48],
    "stem_type": "large",
    "block_configs": [
        [[3, 48, 48, 1, True], [3, 48, 48, 1, False]],
        [[3, 48, 96, 2, False], [3, 96, 96, 1, True], [3, 96, 96, 1, False]],
        [
            [3, 96, 192, 2, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        [
            [3, 192, 384, 2, False],
            [3, 384, 384, 1, True],
            [3, 384, 384, 1, False],
        ],
    ],
}


class PPOCRV6SmallDetConfig(PretrainedConfig):
    model_type = "pp_ocrv6_small_det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", DEFAULT_BACKBONE_CONFIG)
        if isinstance(backbone_config, PPLCNetV4Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = PPLCNetV4Config(**backbone_config)
        else:
            self.backbone_config = PPLCNetV4Config(**DEFAULT_BACKBONE_CONFIG)

        self.reduction = kwargs.get("reduction", 4)
        self.neck_out_channels = kwargs.get("neck_out_channels", 96)
        self.interpolate_mode = kwargs.get("interpolate_mode", "nearest")
        self.kernel_list = kwargs.get("kernel_list", [3, 2, 2])
        self.layer_list_out_channels = kwargs.get(
            "layer_list_out_channels", [48, 96, 192, 384]
        )
        self.dilated_kernel_size = kwargs.get("dilated_kernel_size", 7)
