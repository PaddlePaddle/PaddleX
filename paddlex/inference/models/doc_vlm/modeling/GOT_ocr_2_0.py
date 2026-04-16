# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

"""PP-Chart2Table / GOT-OCR2 model aligned with HuggingFace transformers.

Architecture: GotOcr2VisionEncoder + GotOcr2MultiModalProjector + Qwen2 LM.
Safetensors key names match HF transformers exactly.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.benchmark import add_inference_operations, benchmark
from ...common.transformers.activations import ACT2FN
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ...common.transformers.transformers.model_outputs import CausalLMOutputWithPast
from ._config_pp_chart2table import PPChart2TableConfig
from .qwen2 import Qwen2Model

add_inference_operations("chart2table_generate")


class GotOcr2MLPBlock(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class GotOcr2VisionAttention(nn.Layer):
    """Multi-head attention with optional relative positional embeddings."""

    def __init__(self, config, window_size):
        super().__init__()
        input_size = (
            (
                config.image_size // config.patch_size,
                config.image_size // config.patch_size,
            )
            if window_size == 0
            else (window_size, window_size)
        )

        self.num_attention_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(
            config.hidden_size, config.hidden_size * 3, bias_attr=config.qkv_bias
        )
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = self.create_parameter(
                shape=[2 * input_size[0] - 1, head_dim],
                default_initializer=nn.initializer.Constant(0.0),
            )
            self.rel_pos_w = self.create_parameter(
                shape=[2 * input_size[1] - 1, head_dim],
                default_initializer=nn.initializer.Constant(0.0),
            )

    def get_rel_pos(self, q_size, k_size, rel_pos):
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = F.interpolate(
                rel_pos.reshape([1, rel_pos.shape[0], -1]).transpose([0, 2, 1]),
                size=[max_rel_dist],
                mode="linear",
            )
            rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist]).transpose(
                [1, 0]
            )
        else:
            rel_pos_resized = rel_pos

        q_coords = paddle.arange(q_size).unsqueeze(-1) * max(k_size / q_size, 1.0)
        k_coords = paddle.arange(k_size).unsqueeze(0) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(
            q_size / k_size, 1.0
        )
        return rel_pos_resized[relative_coords.astype("int64")]

    def get_decomposed_rel_pos(self, query, rel_pos_h, rel_pos_w, q_size, k_size):
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = query.shape
        r_q = query.reshape([B, q_h, q_w, dim])
        rel_h = paddle.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = paddle.einsum("bhwc,wkc->bhwk", r_q, Rw)
        return rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]

    def forward(self, hidden_states):
        B, H, W, _ = hidden_states.shape
        qkv = (
            self.qkv(hidden_states)
            .reshape([B, H * W, 3, self.num_attention_heads, -1])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv.reshape([3, B * self.num_attention_heads, H * W, -1]).unbind(
            axis=0
        )

        attn = (q * self.scale) @ k.transpose([0, 2, 1])

        if self.use_rel_pos:
            decomposed = self.get_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
            attn = attn.reshape([B, self.num_attention_heads, H, W, H * W])
            attn = attn + decomposed.reshape([B, self.num_attention_heads, H, W, H * W])
            attn = attn.reshape([B * self.num_attention_heads, H * W, H * W])

        attn = F.softmax(attn, axis=-1)
        x = (
            (attn @ v)
            .reshape([B, self.num_attention_heads, H, W, -1])
            .transpose([0, 2, 3, 1, 4])
            .reshape([B, H, W, -1])
        )
        return self.proj(x)


class GotOcr2VisionLayer(nn.Layer):
    """Transformer block with optional window attention."""

    def __init__(self, config, window_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps
        )
        self.attn = GotOcr2VisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps
        )
        self.mlp = GotOcr2MLPBlock(config)
        self.window_size = window_size

    @staticmethod
    def window_partition(x, window_size):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, pad=[0, 0, 0, pad_w, 0, pad_h], data_format="NHWC")
        Hp, Wp = H + pad_h, W + pad_w
        x = x.reshape(
            [B, Hp // window_size, window_size, Wp // window_size, window_size, C]
        )
        windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape(
            [-1, window_size, window_size, C]
        )
        return windows, (Hp, Wp)

    @staticmethod
    def window_unpartition(windows, window_size, pad_hw, hw):
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.reshape(
            [B, Hp // window_size, Wp // window_size, window_size, window_size, -1]
        )
        x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, Hp, Wp, -1])
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            H, W = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = self.window_partition(
                hidden_states, self.window_size
            )
        hidden_states = self.attn(hidden_states)
        if self.window_size > 0:
            hidden_states = self.window_unpartition(
                hidden_states, self.window_size, pad_hw, (H, W)
            )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class GotOcr2PatchEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        image_size = config.image_size
        patch_size = config.patch_size
        image_size = (
            image_size
            if isinstance(image_size, (tuple, list))
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, (tuple, list))
            else (patch_size, patch_size)
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2D(
            config.num_channels,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, pixel_values):
        embeddings = self.projection(pixel_values)
        # B C H W -> B H W C
        embeddings = embeddings.transpose([0, 2, 3, 1])
        return embeddings


class GotOcr2LayerNorm(nn.LayerNorm):
    """LayerNorm supporting channels_first data format."""

    def __init__(self, normalized_shape, epsilon=1e-6, data_format="channels_last"):
        super().__init__(normalized_shape, epsilon=epsilon)
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channels_first":
            x = x.transpose([0, 2, 3, 1])
            x = super().forward(x)
            x = x.transpose([0, 3, 1, 2])
        else:
            x = super().forward(x)
        return x


class GotOcr2VisionNeck(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2D(
            config.hidden_size, config.output_channels, kernel_size=1, bias_attr=False
        )
        self.layer_norm1 = GotOcr2LayerNorm(
            config.output_channels, data_format="channels_first"
        )
        self.conv2 = nn.Conv2D(
            config.output_channels,
            config.output_channels,
            kernel_size=3,
            padding=1,
            bias_attr=False,
        )
        self.layer_norm2 = GotOcr2LayerNorm(
            config.output_channels, data_format="channels_first"
        )

    def forward(self, hidden_states):
        # B H W C -> B C H W
        hidden_states = hidden_states.transpose([0, 3, 1, 2])
        hidden_states = self.layer_norm1(self.conv1(hidden_states))
        hidden_states = self.layer_norm2(self.conv2(hidden_states))
        return hidden_states


class GotOcr2VisionEncoder(nn.Layer):
    """SAM-ViT based vision encoder for GOT-OCR2."""

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
            layer = GotOcr2VisionLayer(
                config,
                window_size=(
                    config.window_size if i not in config.global_attn_indexes else 0
                ),
            )
            self.layers.append(layer)

        self.neck = GotOcr2VisionNeck(config)

    def forward(self, pixel_values):
        hidden_states = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            hidden_states = hidden_states + self.pos_embed
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # neck outputs [B, C, H, W]
        hidden_states = self.neck(hidden_states)
        return hidden_states


class GotOcr2MultiModalProjector(nn.Layer):
    """Projects vision features to language model dimension."""

    def __init__(self, config):
        super().__init__()
        vision_channels = config.vision_config.output_channels
        language_dim = config.text_config.hidden_size
        self.conv_upsampler1 = nn.Conv2D(
            vision_channels,
            vision_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.conv_upsampler2 = nn.Conv2D(
            vision_channels * 2,
            language_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            bias_attr=False,
        )
        self.multimodal_projector = nn.Linear(language_dim, language_dim)

    def forward(self, vision_embeddings):
        # vision_embeddings: [B, C, H, W]
        hidden = self.conv_upsampler1(vision_embeddings)
        hidden = self.conv_upsampler2(hidden)
        # [B, C, H, W] -> [B, H*W, C]
        hidden = hidden.flatten(2).transpose([0, 2, 1])
        hidden = self.multimodal_projector(hidden)
        return hidden


class GotOcr2Model(nn.Layer):
    """Vision-language model without LM head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = GotOcr2VisionEncoder(config.vision_config)
        self.multi_modal_projector = GotOcr2MultiModalProjector(config)
        self.language_model = Qwen2Model(config.text_config)

    def get_image_features(self, pixel_values):
        vision_output = self.vision_tower(pixel_values)
        image_features = self.multi_modal_projector(vision_output)
        return image_features


class Qwen2LMHead(nn.Layer):
    """LM head that supports weight tying with embeddings via transpose_y."""

    def __init__(self, config, embedding_weights=None, transpose_y=False):
        super().__init__()
        self.transpose_y = transpose_y
        vocab_size = config.vocab_size
        if transpose_y:
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )

    def forward(self, hidden_states):
        return paddle.matmul(hidden_states, self.weight, transpose_y=self.transpose_y)


class PPChart2TableInference(BatchNormHFStateDictMixin, PretrainedModel):
    """PP-Chart2Table inference model aligned with HF transformers GOT-OCR2.

    Hierarchy:
        model.vision_tower      → GotOcr2VisionEncoder
        model.multi_modal_projector → GotOcr2MultiModalProjector
        model.language_model    → Qwen2Model
        lm_head                 → Qwen2LMHead (tied with embed_tokens)
    """

    config_class = PPChart2TableConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = GotOcr2Model(config)

        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(
                config.text_config,
                embedding_weights=self.model.language_model.embed_tokens.weight,
                transpose_y=True,
            )
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config.text_config)

        self.vocab_size = config.text_config.vocab_size
        self.eval()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def get_transpose_weight_keys(self):
        t_layers = [
            # Language model linear layers
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # Vision encoder linear layers
            "attn.qkv",
            "attn.proj",
            "mlp.lin1",
            "mlp.lin2",
            # Multi-modal projector linear layer
            "multimodal_projector",
        ]
        # lm_head uses transpose_y=True so its weight stays in [vocab, hidden] — no transpose
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys

    def _merge_image_features(self, input_ids, pixel_values):
        """Merge vision features into text embeddings at image placeholder positions."""
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)

        if pixel_values is None:
            return inputs_embeds

        # The processor returns images as a list of [1, C, H, W] tensors.
        # Concatenate into a single [B, C, H, W] batch tensor.
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = paddle.concat(pixel_values, axis=0)

        image_features = self.model.get_image_features(
            pixel_values.astype(inputs_embeds.dtype)
        )

        image_token_id = self.config.image_token_index
        batch_size = input_ids.shape[0]

        new_embeds = []
        for i in range(batch_size):
            cur_ids = input_ids[i]
            cur_embeds = inputs_embeds[i]
            cur_features = image_features[i]  # [n_tokens, hidden]

            n_placeholder = int((cur_ids == image_token_id).sum().item())
            n_features = cur_features.shape[0]

            if n_placeholder == 0 and pixel_values is not None:
                raise ValueError(
                    "Image pixels were provided but no image placeholder tokens "
                    f"(id={image_token_id}) found in input_ids."
                )

            if n_placeholder != n_features:
                raise ValueError(
                    f"Number of image placeholder tokens ({n_placeholder}) does not "
                    f"match the number of image features ({n_features}). "
                    "Check that image_seq_length in config matches the "
                    "multi-modal projector output."
                )

            positions = paddle.where(cur_ids == image_token_id)[0].squeeze(-1)
            start_pos = int(positions[0].item())
            end_pos = int(positions[-1].item()) + 1

            merged = paddle.concat(
                [
                    cur_embeds[:start_pos],
                    cur_features,
                    cur_embeds[end_pos:],
                ],
                axis=0,
            )
            new_embeds.append(merged)

        return paddle.stack(new_embeds, axis=0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        images=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # Support both 'pixel_values' and legacy 'images' kwarg
        if pixel_values is None and images is not None:
            pixel_values = images

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.text_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.text_config.output_hidden_states
        )
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.text_config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self._merge_image_features(input_ids, pixel_values)

        outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).astype("float32")

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        batch_size, seq_length = input_ids.shape

        # Always use 2D bool mask — Qwen2 constructs its own 4D causal mask
        attention_mask = paddle.ones((batch_size, seq_length), dtype="bool")

        position_ids = kwargs.get(
            "position_ids",
            paddle.arange(seq_length).expand((batch_size, seq_length)),
        )

        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # Pass images only on first iteration (no KV cache yet)
                "images": kwargs.get("images") if past_key_values is None else None,
            }
        )
        return model_inputs

    @staticmethod
    def update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=False
    ):
        if (
            isinstance(outputs, tuple)
            and len(outputs) > 1
            and not isinstance(outputs[1], paddle.Tensor)
        ):
            model_kwargs["past_key_values"] = outputs[1]
        if isinstance(outputs, CausalLMOutputWithPast) and "past_key_values" in outputs:
            model_kwargs["past_key_values"] = outputs.past_key_values

        if "position_ids" in model_kwargs and model_kwargs["position_ids"] is not None:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = paddle.concat(
                [position_ids, position_ids[..., -1:] + 1], axis=-1
            )

        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if len(attention_mask.shape) == 2:
                model_kwargs["attention_mask"] = paddle.concat(
                    [
                        attention_mask,
                        paddle.ones(
                            [attention_mask.shape[0], 1], dtype=attention_mask.dtype
                        ),
                    ],
                    axis=-1,
                )
        return model_kwargs

    @benchmark.timeit_with_options(name="chart2table_generate")
    def generate(self, inputs, **kwargs):
        """Generate text from image+text inputs.

        Args:
            inputs: dict with 'input_ids' and optionally 'images' tensors.
        """
        input_ids = inputs["input_ids"]
        images = inputs.get("images")
        max_new_tokens = kwargs.pop("max_new_tokens", 1024)
        no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 20)

        with paddle.no_grad():
            generated_ids = super().generate(
                input_ids,
                images=images,
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )

        return generated_ids
