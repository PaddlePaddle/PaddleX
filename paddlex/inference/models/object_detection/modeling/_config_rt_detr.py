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


class RTDETRConfig(PretrainedConfig):
    model_type = "rt_detr"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        backbone_config = kwargs.get("backbone_config", DEFAULT_BACKBONE_CONFIG)
        if isinstance(backbone_config, HGNetV2Config):
            self.backbone_config = backbone_config
        elif isinstance(backbone_config, dict):
            self.backbone_config = HGNetV2Config(**backbone_config)
        else:
            self.backbone_config = HGNetV2Config(**DEFAULT_BACKBONE_CONFIG)

        self.initializer_range = kwargs.get("initializer_range", 0.01)
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-5)
        self.batch_norm_eps = kwargs.get("batch_norm_eps", 1e-5)
        self.freeze_backbone_batch_norms = kwargs.get("freeze_backbone_batch_norms", True)

        # Encoder
        self.encoder_hidden_dim = kwargs.get("encoder_hidden_dim", 256)
        self.encoder_in_channels = kwargs.get("encoder_in_channels", [512, 1024, 2048])
        self.feat_strides = kwargs.get("feat_strides", kwargs.get("feature_strides", [8, 16, 32]))
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

        # Derive num_labels from id2label if present
        id2label = kwargs.get("id2label", None)
        if id2label is not None:
            self.num_labels = len(id2label)
        else:
            self.num_labels = kwargs.get("num_labels", 80)

        self.use_focal_loss = kwargs.get("use_focal_loss", True)
