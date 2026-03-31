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


class SLANeXtVisionConfig(PretrainedConfig):
    """Vision encoder configuration for SLANeXt (same architecture as GOT-OCR2)."""

    model_type = "slanext_vision"

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.output_channels = kwargs.get("output_channels", 256)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.num_channels = kwargs.get("num_channels", 3)
        self.image_size = kwargs.get("image_size", 512)
        self.patch_size = kwargs.get("patch_size", 16)
        self.hidden_act = kwargs.get("hidden_act", "gelu")
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-6)
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.use_abs_pos = kwargs.get("use_abs_pos", True)
        self.use_rel_pos = kwargs.get("use_rel_pos", True)
        self.window_size = kwargs.get("window_size", 14)
        self.global_attn_indexes = kwargs.get("global_attn_indexes", [2, 5, 8, 11])
        self.mlp_dim = kwargs.get("mlp_dim", 3072)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        super().__init__(**kwargs)


class SLANeXtConfig(PretrainedConfig):
    """Configuration for SLANeXt table structure recognition."""

    model_type = "slanext"

    def __init__(self, **kwargs):
        vision_config = kwargs.pop("vision_config", None)
        if vision_config is None:
            self.vision_config = SLANeXtVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = SLANeXtVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.post_conv_in_channels = kwargs.get("post_conv_in_channels", 256)
        self.post_conv_out_channels = kwargs.get("post_conv_out_channels", 512)
        self.out_channels = kwargs.get("out_channels", 50)
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.max_text_length = kwargs.get("max_text_length", 500)
        self.loc_reg_num = kwargs.get("loc_reg_num", 8)
        self.tensor_parallel_degree = 1

        super().__init__(**kwargs)
