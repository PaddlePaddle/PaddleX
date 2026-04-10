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

"""SLANeXt table structure recognition model aligned with HF transformers.

Architecture: GotOcr2VisionEncoder + post_conv backbone + GRU-attention SLA head.
Vision encoder reuses GOT-OCR2 components (identical architecture).
Safetensors key names match HF transformers exactly.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...doc_vlm.modeling.GOT_ocr_2_0 import GotOcr2VisionEncoder
from ._config_slanext import SLANeXtConfig

__all__ = ["SLANeXt"]


class SLANeXtAttentionGRUCell(nn.Layer):
    """Attention-based GRU cell for autoregressive structure decoding."""

    def __init__(self, input_size, hidden_size, num_embeddings):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

    def forward(self, prev_hidden, batch_hidden, char_onehots):
        batch_hidden_proj = self.input_to_hidden(batch_hidden)
        prev_hidden_proj = self.hidden_to_hidden(prev_hidden).unsqueeze(1)

        attention_scores = paddle.tanh(batch_hidden_proj + prev_hidden_proj)
        attention_scores = self.score(attention_scores)

        attn_weights = F.softmax(attention_scores, axis=1)
        attn_weights = attn_weights.transpose([0, 2, 1])
        context = paddle.bmm(attn_weights, batch_hidden).squeeze(1)
        concat_context = paddle.concat([context, char_onehots], axis=1)
        # Paddle GRUCell returns (output, new_states); take output.
        hidden_states, _ = self.rnn(concat_context, prev_hidden)

        return hidden_states, attn_weights


class SLANeXtMLP(nn.Layer):
    """Two-layer MLP for structure token prediction."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class SLANeXtBackbone(nn.Layer):
    """Vision backbone: SAM-ViT encoder + post-conv downsampling."""

    def __init__(self, config):
        super().__init__()
        self.vision_tower = GotOcr2VisionEncoder(config.vision_config)
        self.post_conv = nn.Conv2D(
            config.post_conv_in_channels, config.post_conv_out_channels,
            kernel_size=3, stride=2, padding=1, bias_attr=False,
        )

    def forward(self, pixel_values):
        # vision_tower returns [B, C, H, W]
        hidden_states = self.vision_tower(pixel_values)
        hidden_states = self.post_conv(hidden_states)
        # [B, C, H, W] -> [B, H*W, C]
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])
        return hidden_states


class SLANeXtSLAHead(nn.Layer):
    """Autoregressive SLA head for table structure prediction."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.structure_attention_cell = SLANeXtAttentionGRUCell(
            config.post_conv_out_channels, config.hidden_size, config.out_channels,
        )
        self.structure_generator = SLANeXtMLP(config.hidden_size, config.out_channels)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        features = paddle.zeros([batch_size, self.config.hidden_size], dtype="float32")
        predicted_chars = paddle.zeros([batch_size], dtype="int64")

        structure_preds_list = []
        structure_ids_list = []
        for _ in range(self.config.max_text_length + 1):
            embedding_feature = F.one_hot(
                predicted_chars, self.config.out_channels
            ).astype("float32")
            features, _ = self.structure_attention_cell(
                features,
                hidden_states.astype("float32"),
                embedding_feature,
            )
            structure_step = self.structure_generator(features)
            predicted_chars = structure_step.argmax(axis=1)

            structure_preds_list.append(structure_step)
            structure_ids_list.append(predicted_chars)
            if (
                paddle.stack(structure_ids_list, axis=1)
                .equal(paddle.to_tensor(self.config.out_channels - 1))
                .any(axis=-1)
                .all()
            ):
                break

        structure_preds = paddle.stack(structure_preds_list, axis=1)
        structure_probs = F.softmax(structure_preds, axis=-1)
        return structure_probs


class SLANeXt(BatchNormHFStateDictMixin, PretrainedModel):
    """SLANeXt table structure recognition model.

    Hierarchy:
        backbone.vision_tower  → GotOcr2VisionEncoder (shared with GOT-OCR2)
        backbone.post_conv     → Conv2D stride-2 downsampling
        head.structure_attention_cell → SLANeXtAttentionGRUCell
        head.structure_generator     → SLANeXtMLP
    """

    config_class = SLANeXtConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.backbone = SLANeXtBackbone(config)
        self.head = SLANeXtSLAHead(config)

    def forward(self, x):
        pixel_values = paddle.to_tensor(x[0])

        # Handle 1-channel input by expanding to 3 channels
        if pixel_values.shape[1] == 1:
            pixel_values = paddle.expand(pixel_values, [-1, 3, -1, -1])

        features = self.backbone(pixel_values)
        structure_probs = self.head(features)

        # Return [loc_preds, structure_probs] for backward compatibility
        # with the predictor/postprocessor pipeline.
        # HF model doesn't predict locations; fill with zeros.
        loc_preds = paddle.zeros(
            [
                structure_probs.shape[0],
                structure_probs.shape[1],
                self.config.loc_reg_num,
            ],
            dtype=structure_probs.dtype,
        )
        return [loc_preds, structure_probs]

    def get_transpose_weight_keys(self):
        t_layers = [
            "attn.qkv",
            "attn.proj",
            "mlp.lin1",
            "mlp.lin2",
            "structure_attention_cell.score",
            "structure_attention_cell.input_to_hidden",
            "structure_attention_cell.hidden_to_hidden",
            "structure_generator.fc1",
            "structure_generator.fc2",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys
