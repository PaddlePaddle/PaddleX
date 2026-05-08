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

DEFAULT_BACKBONE_CONFIG = {
    "model_type": "uvdoc_backbone",
    "resnet_head": [[3, 32], [32, 32]],
    "resnet_configs": [
        [[32, 32, 1, False], [32, 32, 3, False], [32, 32, 3, False]],
        [
            [32, 64, 1, True],
            [64, 64, 3, False],
            [64, 64, 3, False],
            [64, 64, 3, False],
        ],
        [
            [64, 128, 1, True],
            [128, 128, 3, False],
            [128, 128, 3, False],
            [128, 128, 3, False],
            [128, 128, 3, False],
            [128, 128, 3, False],
        ],
    ],
    "stage_configs": [
        [[128, 1]],
        [[128, 2]],
        [[128, 5]],
        [[128, 8], [128, 3], [128, 2]],
        [[128, 12], [128, 7], [128, 4]],
        [[128, 18], [128, 12], [128, 6]],
    ],
    "kernel_size": 5,
}

DEFAULT_CONFIG = {
    "model_name": "UVDoc",
    "hidden_act": "prelu",
    "padding_mode": "reflect",
    "kernel_size": 5,
    "bridge_connector": [128, 128],
    "out_point_positions2D": [[128, 32], [32, 2]],
    "upsample_size": [712, 488],
    "upsample_mode": "bilinear",
}


class UVDocBackboneConfig(PretrainedConfig):
    model_type = "uvdoc_backbone"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.resnet_head = kwargs.get(
            "resnet_head", DEFAULT_BACKBONE_CONFIG["resnet_head"]
        )
        self.resnet_configs = kwargs.get(
            "resnet_configs", DEFAULT_BACKBONE_CONFIG["resnet_configs"]
        )
        self.stage_configs = kwargs.get(
            "stage_configs", DEFAULT_BACKBONE_CONFIG["stage_configs"]
        )
        self.kernel_size = kwargs.get(
            "kernel_size", DEFAULT_BACKBONE_CONFIG["kernel_size"]
        )


class UVDocConfig(PretrainedConfig):
    model_type = "uvdoc"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs.get("model_name", DEFAULT_CONFIG["model_name"])
        self.hidden_act = kwargs.get("hidden_act", DEFAULT_CONFIG["hidden_act"])
        self.padding_mode = kwargs.get("padding_mode", DEFAULT_CONFIG["padding_mode"])
        self.kernel_size = kwargs.get("kernel_size", DEFAULT_CONFIG["kernel_size"])
        self.bridge_connector = kwargs.get(
            "bridge_connector", DEFAULT_CONFIG["bridge_connector"]
        )
        self.out_point_positions2D = kwargs.get(
            "out_point_positions2D", DEFAULT_CONFIG["out_point_positions2D"]
        )
        self.upsample_size = kwargs.get(
            "upsample_size", DEFAULT_CONFIG["upsample_size"]
        )
        self.upsample_mode = kwargs.get(
            "upsample_mode", DEFAULT_CONFIG["upsample_mode"]
        )

        backbone_config = kwargs.get("backbone_config", DEFAULT_BACKBONE_CONFIG)
        if isinstance(backbone_config, dict):
            self.backbone_config = UVDocBackboneConfig(**backbone_config)
        elif isinstance(backbone_config, UVDocBackboneConfig):
            self.backbone_config = backbone_config
        else:
            self.backbone_config = UVDocBackboneConfig(**DEFAULT_BACKBONE_CONFIG)
