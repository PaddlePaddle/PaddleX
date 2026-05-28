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


class PPOCRV6TinyRecConfig(PretrainedConfig):
    model_type = "pp_ocrv6_tiny_rec"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", None)
        if isinstance(backbone_config, PPLCNetV4Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = PPLCNetV4Config(**backbone_config)
        else:
            self.backbone_config = PPLCNetV4Config()

        self.hidden_size = kwargs.get("hidden_size", 120)
        self.head_out_channels = kwargs.get("head_out_channels", 6625)
