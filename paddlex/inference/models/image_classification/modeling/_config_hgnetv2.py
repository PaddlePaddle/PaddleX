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

DEFAULT_HGNETV2_CONFIG = {
    "num_channels": 3,
    "embedding_size": 64,
    "hidden_sizes": [256, 512, 1024, 2048],
    "hidden_act": "relu",
    "num_labels": 1000,
    "stem_channels": [3, 32, 48],
    "stem_strides": [2, 1, 1, 2, 1],
    "stage_in_channels": [48, 128, 512, 1024],
    "stage_mid_channels": [48, 96, 192, 384],
    "stage_out_channels": [128, 512, 1024, 2048],
    "stage_num_blocks": [1, 1, 3, 1],
    "stage_downsample": [False, True, True, True],
    "stage_downsample_strides": [2, 2, 2, 2],
    "stage_light_block": [False, False, True, True],
    "stage_kernel_size": [3, 3, 5, 5],
    "stage_numb_of_layers": [6, 6, 6, 6],
    "use_learnable_affine_block": False,
}


class HGNetV2Config(PretrainedConfig):
    model_type = "hgnet_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = kwargs.get(
            "num_channels", DEFAULT_HGNETV2_CONFIG["num_channels"]
        )
        self.embedding_size = kwargs.get(
            "embedding_size", DEFAULT_HGNETV2_CONFIG["embedding_size"]
        )
        self.hidden_sizes = kwargs.get(
            "hidden_sizes", DEFAULT_HGNETV2_CONFIG["hidden_sizes"]
        )
        self.hidden_act = kwargs.get(
            "hidden_act", DEFAULT_HGNETV2_CONFIG["hidden_act"]
        )
        self.num_labels = kwargs.get(
            "num_labels", DEFAULT_HGNETV2_CONFIG["num_labels"]
        )
        self.stem_channels = kwargs.get(
            "stem_channels", DEFAULT_HGNETV2_CONFIG["stem_channels"]
        )
        self.stem_strides = kwargs.get(
            "stem_strides", DEFAULT_HGNETV2_CONFIG["stem_strides"]
        )
        self.stage_in_channels = kwargs.get(
            "stage_in_channels", DEFAULT_HGNETV2_CONFIG["stage_in_channels"]
        )
        self.stage_mid_channels = kwargs.get(
            "stage_mid_channels", DEFAULT_HGNETV2_CONFIG["stage_mid_channels"]
        )
        self.stage_out_channels = kwargs.get(
            "stage_out_channels", DEFAULT_HGNETV2_CONFIG["stage_out_channels"]
        )
        self.stage_num_blocks = kwargs.get(
            "stage_num_blocks", DEFAULT_HGNETV2_CONFIG["stage_num_blocks"]
        )
        self.stage_downsample = kwargs.get(
            "stage_downsample", DEFAULT_HGNETV2_CONFIG["stage_downsample"]
        )
        self.stage_downsample_strides = kwargs.get(
            "stage_downsample_strides",
            DEFAULT_HGNETV2_CONFIG["stage_downsample_strides"],
        )
        self.stage_light_block = kwargs.get(
            "stage_light_block", DEFAULT_HGNETV2_CONFIG["stage_light_block"]
        )
        self.stage_kernel_size = kwargs.get(
            "stage_kernel_size", DEFAULT_HGNETV2_CONFIG["stage_kernel_size"]
        )
        self.stage_numb_of_layers = kwargs.get(
            "stage_numb_of_layers", DEFAULT_HGNETV2_CONFIG["stage_numb_of_layers"]
        )
        self.use_learnable_affine_block = kwargs.get(
            "use_learnable_affine_block",
            DEFAULT_HGNETV2_CONFIG["use_learnable_affine_block"],
        )
