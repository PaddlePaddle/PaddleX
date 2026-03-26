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
from ...image_classification.modeling._config_pplcnetv3 import PPLCNetV3Config

DEFAULT_BACKBONE_CONFIG = {
    "model_type": "pp_lcnet_v3",
    "scale": 0.75,
    "divisor": 16,
}


class PPOCRV5MobileDetConfig(PretrainedConfig):
    model_type = "pp_ocrv5_mobile_det"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", DEFAULT_BACKBONE_CONFIG)
        if isinstance(backbone_config, PPLCNetV3Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = PPLCNetV3Config(**backbone_config)
        else:
            self.backbone_config = PPLCNetV3Config(**DEFAULT_BACKBONE_CONFIG)

        self.reduction = kwargs.get("reduction", 4)
        self.neck_out_channels = kwargs.get("neck_out_channels", 96)
        self.interpolate_mode = kwargs.get("interpolate_mode", "nearest")
        self.kernel_list = kwargs.get("kernel_list", [3, 2, 2])
        self.layer_list_out_channels = kwargs.get(
            "layer_list_out_channels", [12, 18, 42, 360]
        )
