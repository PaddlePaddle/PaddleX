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
from ...image_classification.modeling._config_hgnetv2 import HGNetV2Config

DEFAULT_BACKBONE_CONFIG = {
    "model_type": "hgnet_v2",
    "num_channels": 3,
    "embedding_size": 64,
    "hidden_sizes": [256, 512, 1024, 2048],
    "hidden_act": "relu",
    "num_labels": 0,
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


class PPDocLayoutV2ReadingOrderConfig(PretrainedConfig):
    model_type = "pp_doclayout_v2_reading_order"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.attention_probs_dropout_prob = kwargs.get("attention_probs_dropout_prob", 0.1)
        self.has_relative_attention_bias = kwargs.get("has_relative_attention_bias", False)
        self.has_spatial_attention_bias = kwargs.get("has_spatial_attention_bias", True)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-5)
        self.hidden_dropout_prob = kwargs.get("hidden_dropout_prob", 0.1)
        self.intermediate_size = kwargs.get("intermediate_size", 2048)
        self.hidden_act = kwargs.get("hidden_act", "gelu")
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 6)
        self.rel_pos_bins = kwargs.get("rel_pos_bins", 32)
        self.max_rel_pos = kwargs.get("max_rel_pos", 128)
        self.rel_2d_pos_bins = kwargs.get("rel_2d_pos_bins", 64)
        self.max_rel_2d_pos = kwargs.get("max_rel_2d_pos", 256)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 514)
        self.max_2d_position_embeddings = kwargs.get("max_2d_position_embeddings", 1024)
        self.type_vocab_size = kwargs.get("type_vocab_size", 1)
        self.vocab_size = kwargs.get("vocab_size", 4)
        self.initializer_range = kwargs.get("initializer_range", 0.01)
        self.start_token_id = kwargs.get("start_token_id", 0)
        self.pad_token_id = kwargs.get("pad_token_id", 1)
        self.end_token_id = kwargs.get("end_token_id", 2)
        self.pred_token_id = kwargs.get("pred_token_id", 3)
        self.coordinate_size = kwargs.get("coordinate_size", 171)
        self.shape_size = kwargs.get("shape_size", 170)
        self.num_classes = kwargs.get("num_classes", 20)
        self.relation_bias_embed_dim = kwargs.get("relation_bias_embed_dim", 16)
        self.relation_bias_theta = kwargs.get("relation_bias_theta", 10000)
        self.relation_bias_scale = kwargs.get("relation_bias_scale", 100)
        self.global_pointer_head_size = kwargs.get("global_pointer_head_size", 64)
        self.gp_dropout_value = kwargs.get("gp_dropout_value", 0.0)
        self.chunk_size_feed_forward = kwargs.get("chunk_size_feed_forward", 0)


class PPDocLayoutV2Config(PretrainedConfig):
    model_type = "pp_doclayout_v2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", DEFAULT_BACKBONE_CONFIG)
        if isinstance(backbone_config, HGNetV2Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = HGNetV2Config(**backbone_config)
        else:
            self.backbone_config = HGNetV2Config(**DEFAULT_BACKBONE_CONFIG)

        reading_order_config = kwargs.get("reading_order_config", {})
        if isinstance(reading_order_config, PPDocLayoutV2ReadingOrderConfig):
            self.reading_order_config = reading_order_config
        elif isinstance(reading_order_config, dict):
            self.reading_order_config = PPDocLayoutV2ReadingOrderConfig(**reading_order_config)
        else:
            self.reading_order_config = PPDocLayoutV2ReadingOrderConfig()

        self.initializer_range = kwargs.get("initializer_range", 0.01)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-5)
        self.batch_norm_eps = kwargs.get("batch_norm_eps", 1e-5)
        self.freeze_backbone_batch_norms = kwargs.get("freeze_backbone_batch_norms", True)

        # Encoder
        self.encoder_hidden_dim = kwargs.get("encoder_hidden_dim", 256)
        self.encoder_in_channels = kwargs.get("encoder_in_channels", [512, 1024, 2048])
        self.feat_strides = kwargs.get("feat_strides", [8, 16, 32])
        self.encoder_layers = kwargs.get("encoder_layers", 1)
        self.encoder_ffn_dim = kwargs.get("encoder_ffn_dim", 1024)
        self.encoder_attention_heads = kwargs.get("encoder_attention_heads", 8)
        self.num_attention_heads = kwargs.get("num_attention_heads", self.encoder_attention_heads)
        self.dropout = kwargs.get("dropout", 0.0)
        self.activation_dropout = kwargs.get("activation_dropout", 0.0)
        self.encode_proj_layers = kwargs.get("encode_proj_layers", [2])
        self.positional_encoding_temperature = kwargs.get("positional_encoding_temperature", 10000)
        self.encoder_activation_function = kwargs.get("encoder_activation_function", "gelu")
        self.activation_function = kwargs.get("activation_function", "silu")
        self.eval_size = kwargs.get("eval_size", None)
        self.normalize_before = kwargs.get("normalize_before", False)
        self.hidden_expansion = kwargs.get("hidden_expansion", 1.0)

        # Decoder
        self.d_model = kwargs.get("d_model", 256)
        self.num_queries = kwargs.get("num_queries", 300)
        self.decoder_in_channels = kwargs.get("decoder_in_channels", [256, 256, 256])
        self.decoder_ffn_dim = kwargs.get("decoder_ffn_dim", 1024)
        self.num_feature_levels = kwargs.get("num_feature_levels", 3)
        self.decoder_n_points = kwargs.get("decoder_n_points", 4)
        self.decoder_layers = kwargs.get("decoder_layers", 6)
        self.decoder_attention_heads = kwargs.get("decoder_attention_heads", 8)
        self.decoder_activation_function = kwargs.get("decoder_activation_function", "relu")
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)

        # Denoising
        self.num_denoising = kwargs.get("num_denoising", 100)
        self.label_noise_ratio = kwargs.get("label_noise_ratio", 0.5)
        self.box_noise_scale = kwargs.get("box_noise_scale", 1.0)
        self.learn_initial_query = kwargs.get("learn_initial_query", False)
        self.anchor_image_size = kwargs.get("anchor_image_size", None)
        self.disable_custom_kernels = kwargs.get("disable_custom_kernels", True)
        self.num_labels = kwargs.get("num_labels", 25)
        self.initializer_bias_prior_prob = kwargs.get("initializer_bias_prior_prob", None)

        # Post-processing
        self.class_thresholds = kwargs.get("class_thresholds", None)
        self.class_order = kwargs.get("class_order", None)
