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
    "model_name": "UVDoc",
    "num_filter": 32,
    "in_channels": 3,
    "kernel_size": 5,
    "stride": [1, 2, 2, 2],
    "map_num": [1, 2, 4, 8, 16],
    "block_nums": [3, 4, 6, 3],
    "dilation_values": {
        "bridge_1": 1,
        "bridge_2": 2,
        "bridge_3": 5,
        "bridge_4": [8, 3, 2],
        "bridge_5": [12, 7, 4],
        "bridge_6": [18, 12, 6],
    },
    "padding_mode": "reflect",
    "upsample_size": [712, 488],
    "upsample_mode": "bilinear",
}


class UVDocNetConfig(PretrainedConfig):
    model_type = "uvdoc"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.model_name = kwargs.get("model_name", DEFAULT_CONFIG["model_name"])
        self.num_filter = kwargs.get("num_filter", DEFAULT_CONFIG["num_filter"])
        self.in_channels = kwargs.get("in_channels", DEFAULT_CONFIG["in_channels"])
        self.kernel_size = kwargs.get("kernel_size", DEFAULT_CONFIG["kernel_size"])
        self.stride = kwargs.get("stride", DEFAULT_CONFIG["stride"])
        self.map_num = kwargs.get("map_num", DEFAULT_CONFIG["map_num"])
        self.block_nums = kwargs.get("block_nums", DEFAULT_CONFIG["block_nums"])
        self.dilation_values = kwargs.get(
            "dilation_values", DEFAULT_CONFIG["dilation_values"]
        )
        self.padding_mode = kwargs.get("padding_mode", DEFAULT_CONFIG["padding_mode"])
        self.upsample_size = kwargs.get(
            "upsample_size", DEFAULT_CONFIG["upsample_size"]
        )
        self.upsample_mode = kwargs.get(
            "upsample_mode", DEFAULT_CONFIG["upsample_mode"]
        )
