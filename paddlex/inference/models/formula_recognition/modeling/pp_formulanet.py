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

"""PP-FormulaNet aligned with HF transformers (paddle_dynamic engine).

Architecture: SAM ViT-B encoder (shared with GOT-OCR2 / SLANeXt) + multi-modal
projector (2 conv + 2 linear) + MBart decoder + LM head.
Safetensors key names match ``transformers.models.pp_formulanet`` exactly.
"""

import paddle
import paddle.nn as nn

from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...doc_vlm.modeling.GOT_ocr_2_0 import (
    GotOcr2PatchEmbeddings,
    GotOcr2VisionLayer,
    GotOcr2VisionNeck,
)
from ._config_pp_formulanet import PPFormulaNetConfig
from .mbart import MBartDecoder

__all__ = ["PPFormulaNet"]


class PPFormulaNetMultiModalProjector(nn.Layer):
    """Two stride-2 convs to downsample, then two linears to project.

    Mirrors ``transformers.models.pp_formulanet.PPFormulaNetMultiModalProjector``.
    Input:  vision features [B, post_conv_in_channels, H, W]
    Output: token sequence  [B, H/4 * W/4, decoder_hidden_size]
    """

    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2D(
            config.post_conv_in_channels,
            config.post_conv_mid_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.conv2 = nn.Conv2D(
            config.post_conv_mid_channels,
            config.post_conv_out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.linear_1 = nn.Linear(
            config.post_conv_out_channels, config.post_conv_out_channels
        )
        self.linear_2 = nn.Linear(
            config.post_conv_out_channels, config.decoder_hidden_size
        )

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        # [B, C, H, W] -> [B, H*W, C]
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PPFormulaNetVisionModel(nn.Layer):
    """SAM ViT-B encoder + multi-modal projector.

    Reuses GOT-OCR2 SAM-ViT components (identical architecture); the projector
    is held under ``multi_modal_projector.*`` to match HF layout
    ``model.encoder.multi_modal_projector.*``.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embed = GotOcr2PatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = self.create_parameter(
                shape=[
                    1,
                    config.image_size // config.patch_size,
                    config.image_size // config.patch_size,
                    config.hidden_size,
                ],
                default_initializer=nn.initializer.Constant(0.0),
            )

        self.layers = nn.LayerList()
        for i in range(config.num_hidden_layers):
            window_size = (
                config.window_size if i not in config.global_attn_indexes else 0
            )
            self.layers.append(GotOcr2VisionLayer(config, window_size=window_size))

        self.neck = GotOcr2VisionNeck(config)
        self.multi_modal_projector = PPFormulaNetMultiModalProjector(config)

    def forward(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.neck(hidden_states)
        # [B, C, H, W] -> [B, T, d_model]
        return self.multi_modal_projector(hidden_states)


class PPFormulaNetModel(nn.Layer):
    """Encoder-decoder pair without the LM head, mirroring HF ``PPFormulaNetModel``."""

    def __init__(self, config: PPFormulaNetConfig):
        super().__init__()
        self.config = config
        self.encoder = PPFormulaNetVisionModel(config.vision_config)
        self.decoder = MBartDecoder(config.text_config)


class PPFormulaNet(BatchNormHFStateDictMixin, PretrainedModel):
    """PP-FormulaNet for conditional formula generation.

    Hierarchy (matches HF safetensors):
        model.encoder.patch_embed / pos_embed / layers / neck / multi_modal_projector
        model.decoder.embed_tokens / embed_positions / layers / layer_norm
        lm_head
    """

    config_class = PPFormulaNetConfig

    def __init__(self, config: PPFormulaNetConfig):
        super().__init__(config)
        self.config = config
        self.model = PPFormulaNetModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias_attr=False,
        )

    def get_transpose_weight_keys(self):
        # Linear-layer weights need transpose when loading HF safetensors
        # (HF stores [out, in]; Paddle nn.Linear stores [in, out]).
        t_layers = [
            # Vision encoder linear layers (SAM ViT-B)
            "attn.qkv",
            "attn.proj",
            "mlp.lin1",
            "mlp.lin2",
            # Multi-modal projector linears
            "multi_modal_projector.linear_1",
            "multi_modal_projector.linear_2",
            # MBart decoder linear layers
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "encoder_attn.q_proj",
            "encoder_attn.k_proj",
            "encoder_attn.v_proj",
            "encoder_attn.out_proj",
            "fc1",
            "fc2",
            # LM head
            "lm_head",
        ]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys

    def _encode(self, pixel_values: paddle.Tensor) -> paddle.Tensor:
        if pixel_values.shape[1] == 1:
            pixel_values = paddle.expand(pixel_values, [-1, 3, -1, -1])
        return self.model.encoder(pixel_values)

    def _decode_step(
        self,
        input_ids: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        cache: list,
        past_length: int,
    ) -> paddle.Tensor:
        hidden_states = self.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            cache=cache,
            past_length=past_length,
        )
        return self.lm_head(hidden_states)

    @paddle.no_grad()
    def forward(self, x):
        """Greedy autoregressive decoding for inference.

        Args:
            x: list/tuple where ``x[0]`` is a [B, 1 or 3, H, W] image batch.

        Returns:
            ``[token_ids]``: list with one [B, T] int64 tensor of decoded ids.
            ``T`` is determined per-batch — generation stops when every
            sequence has emitted EOS or ``max_position_embeddings`` is hit.
            Sequences finished early are padded with ``pad_token_id``.
        """
        text_config = self.config.text_config
        eos_token_id = text_config.eos_token_id
        pad_token_id = text_config.pad_token_id
        decoder_start_token_id = text_config.decoder_start_token_id
        # Cap by max_position_embeddings; the +2 offset on positional embeds
        # already buys us 2 extra rows but the logical length is the config value.
        max_length = text_config.max_position_embeddings

        pixel_values = paddle.to_tensor(x[0]) if not isinstance(x[0], paddle.Tensor) else x[0]
        encoder_hidden_states = self._encode(pixel_values)
        batch_size = encoder_hidden_states.shape[0]

        cache = [{} for _ in range(text_config.decoder_layers)]
        # Seed with the start token. We do NOT keep the start token in output —
        # in HF PP-FormulaNet decoder_start_token_id == eos_token_id == 2, so
        # leaving it in would make the downstream UniMERNetDecode (which
        # truncates at the first EOS) return empty strings.
        input_ids = paddle.full(
            [batch_size, 1], decoder_start_token_id, dtype="int64"
        )
        generated = []
        unfinished = paddle.ones([batch_size], dtype="int64")

        past_length = 0
        for _ in range(max_length - 1):
            logits = self._decode_step(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                cache=cache,
                past_length=past_length,
            )
            past_length += input_ids.shape[1]
            next_logits = logits[:, -1, :]
            next_tokens = paddle.argmax(next_logits, axis=-1)
            # Pad finished sequences.
            next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)
            input_ids = next_tokens.unsqueeze(-1)
            generated.append(input_ids)

            unfinished = unfinished & (next_tokens != eos_token_id).astype("int64")
            if int(unfinished.sum().item()) == 0:
                break

        return [paddle.concat(generated, axis=-1)]
