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


class PPFormulaNetVisionConfig(PretrainedConfig):
    """Vision encoder configuration for PP-FormulaNet (SAM ViT-B + multi-modal projector).

    Mirrors ``transformers.models.pp_formulanet.PPFormulaNetVisionConfig``.
    """

    model_type = "pp_formulanet_vision"

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
        # Multi-modal projector
        self.post_conv_in_channels = kwargs.get("post_conv_in_channels", 256)
        self.post_conv_mid_channels = kwargs.get("post_conv_mid_channels", 512)
        self.post_conv_out_channels = kwargs.get("post_conv_out_channels", 1024)
        self.decoder_hidden_size = kwargs.get("decoder_hidden_size", 512)
        super().__init__(**kwargs)


class PPFormulaNetTextConfig(PretrainedConfig):
    """Text decoder configuration for PP-FormulaNet (MBart-style decoder).

    Mirrors ``transformers.models.pp_formulanet.PPFormulaNetTextConfig``.
    """

    model_type = "pp_formulanet_text"

    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 50000)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2560)
        self.d_model = kwargs.get("d_model", kwargs.get("hidden_size", 512))
        self.hidden_size = self.d_model
        self.encoder_layers = kwargs.get("encoder_layers", 12)
        self.encoder_attention_heads = kwargs.get("encoder_attention_heads", 16)
        self.decoder_layers = kwargs.get("decoder_layers", 8)
        self.decoder_ffn_dim = kwargs.get("decoder_ffn_dim", 2048)
        self.decoder_attention_heads = kwargs.get("decoder_attention_heads", 16)
        self.decoder_layerdrop = kwargs.get("decoder_layerdrop", 0.0)
        self.activation_function = kwargs.get("activation_function", "gelu")
        self.dropout = kwargs.get("dropout", 0.1)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.activation_dropout = kwargs.get("activation_dropout", 0.0)
        self.init_std = kwargs.get("init_std", 0.02)
        self.scale_embedding = kwargs.get("scale_embedding", True)
        self.use_cache = kwargs.get("use_cache", True)
        self.is_encoder_decoder = kwargs.get("is_encoder_decoder", True)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.num_attention_heads = self.encoder_attention_heads
        self.num_hidden_layers = self.encoder_layers
        # Special token IDs are pop'd by PretrainedConfig.__init__ — fill in
        # defaults via kwargs so that super(...) sets them correctly instead of
        # overwriting our values with None.
        kwargs.setdefault("pad_token_id", 1)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 2)
        kwargs.setdefault("decoder_start_token_id", 2)
        kwargs.setdefault("forced_eos_token_id", 2)
        super().__init__(**kwargs)


class PPFormulaNetConfig(PretrainedConfig):
    """Configuration for PP-FormulaNet conditional generation."""

    model_type = "pp_formulanet"

    def __init__(self, **kwargs):
        text_config = kwargs.pop("text_config", None)
        if text_config is None:
            self.text_config = PPFormulaNetTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = PPFormulaNetTextConfig(**text_config)
        else:
            self.text_config = text_config

        vision_config = kwargs.pop("vision_config", None)
        if vision_config is None:
            self.vision_config = PPFormulaNetVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = PPFormulaNetVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        # Keep decoder hidden size in sync between text and vision sides — the
        # multi-modal projector's linear_2 must project encoder features to the
        # decoder's d_model.
        self.vision_config.decoder_hidden_size = self.text_config.d_model

        self.is_encoder_decoder = kwargs.get("is_encoder_decoder", True)
        self.tensor_parallel_degree = 1

        super().__init__(**kwargs)
