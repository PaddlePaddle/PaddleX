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
from .qwen2 import Qwen2Config


class GotOcr2VisionConfig(PretrainedConfig):
    """Vision encoder configuration for GOT-OCR2 / PP-Chart2Table."""

    model_type = "got_ocr2_vision"

    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.output_channels = kwargs.get("output_channels", 256)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.num_channels = kwargs.get("num_channels", 3)
        self.image_size = kwargs.get("image_size", 1024)
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


class PPChart2TableConfig(PretrainedConfig):
    """Configuration for PP-Chart2Table (GOT-OCR2 architecture)."""

    model_type = "pp_chart2table"

    def __init__(self, **kwargs):
        vision_config = kwargs.pop("vision_config", None)
        if vision_config is None:
            self.vision_config = GotOcr2VisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = GotOcr2VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        text_config = kwargs.pop("text_config", None)
        if text_config is None:
            self.text_config = Qwen2Config(
                vocab_size=151860,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=16,
                hidden_act="silu",
                max_position_embeddings=32768,
                rms_norm_eps=1e-6,
                rope_theta=1000000.0,
            )
        elif isinstance(text_config, dict):
            # Extract rope_theta from nested rope_parameters if present
            rope_params = text_config.pop("rope_parameters", None)
            if rope_params and "rope_theta" not in text_config:
                text_config["rope_theta"] = rope_params.get("rope_theta", 10000.0)
            # Remove fields unknown to Qwen2Config
            text_config.pop("architectures", None)
            text_config.pop("auto_map", None)
            text_config.pop("model_type", None)
            self.text_config = Qwen2Config(**text_config)
        else:
            self.text_config = text_config

        self.image_token_index = kwargs.get("image_token_index", 151859)
        self.image_seq_length = kwargs.get("image_seq_length", 256)

        tie = kwargs.get("tie_word_embeddings", True)
        self.tie_word_embeddings = tie
        self.text_config.tie_word_embeddings = tie

        # Pass vocab_size down to text config if specified at top level
        top_vocab = kwargs.get("vocab_size", None)
        if top_vocab is not None:
            self.text_config.vocab_size = top_vocab

        super().__init__(**kwargs)
