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

from functools import partial
from typing import List, Optional, Tuple, Type

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ...common.vlm.transformers.model_outputs import CausalLMOutputWithPast
from .qwen2 import Qwen2Config, Qwen2ForCausalLM, Qwen2Model


class MLPBlock(paddle.nn.Layer):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[paddle.nn.Layer] = paddle.nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(paddle.nn.Layer):
    def __init__(self, num_channels: int, epsilon: float = 1e-06) -> None:
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.ones(shape=num_channels)
        )
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=num_channels)
        )
        self.epsilon = epsilon

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        u = x.mean(axis=1, keepdim=True)
        s = (x - u).pow(y=2).mean(axis=1, keepdim=True)
        x = (x - u) / paddle.sqrt(x=s + self.epsilon)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ImageEncoderViT(paddle.nn.Layer):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Layer] = nn.LayerNorm,
        act_layer: Type[nn.Layer] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Layer): Normalization layer.
            act_layer (nn.Layer): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[paddle.base.framework.EagerParamBase.from_tensor] = (
            None
        )
        if use_abs_pos:
            self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(
                    shape=[1, img_size // patch_size, img_size // patch_size, embed_dim]
                )
            )

        self.blocks = paddle.nn.LayerList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2D(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias_attr=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2D(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias_attr=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.net_2 = nn.Conv2D(
            256, 512, kernel_size=3, stride=2, padding=1, bias_attr=False
        )
        self.net_3 = nn.Conv2D(
            512, 1024, kernel_size=3, stride=2, padding=1, bias_attr=False
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.transpose([0, 3, 1, 2]))
        x = self.net_2(x)
        x = self.net_3(x)
        return x


class Block(paddle.nn.Layer):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Layer] = nn.LayerNorm,
        act_layer: Type[nn.Layer] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Layer): Normalization layer.
            act_layer (nn.Layer): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x


class Attention(paddle.nn.Layer):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            self.rel_pos_h = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[2 * input_size[0] - 1, head_dim])
            )
            self.rel_pos_w = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[2 * input_size[1] - 1, head_dim])
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, H, W, _ = tuple(x.shape)
        qkv = (
            self.qkv(x)
            .reshape([B, H * W, 3, self.num_heads, -1])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv.reshape([3, B * self.num_heads, H * W, -1]).unbind(axis=0)

        attn = (q * self.scale) @ k.transpose([0, 2, 1])

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = F.softmax(attn, axis=-1)
        x = (
            (attn @ v)
            .reshape([B, self.num_heads, H, W, -1])
            .transpose([0, 2, 3, 1, 4])
            .reshape([B, H, W, -1])
        )
        x = self.proj(x)

        return x


def window_partition(
    x: paddle.Tensor, window_size: int
) -> Tuple[paddle.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = tuple(x.shape)

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, pad=(0, pad_w, 0, pad_h), data_format="NHWC")
    Hp, Wp = H + pad_h, W + pad_w

    x = x.reshape(
        [B, Hp // window_size, window_size, Wp // window_size, window_size, C]
    )
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows, (Hp, Wp)


def window_unpartition(
    windows: paddle.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> paddle.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = tuple(windows.shape)[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(
        [B, Hp // window_size, Wp // window_size, window_size, window_size, -1]
    )
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, Hp, Wp, -1])
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: paddle.Tensor) -> paddle.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if tuple(rel_pos.shape)[0] != max_rel_dist:
        rel_pos_resized = paddle.nn.functional.interpolate(
            rel_pos.reshape([1, tuple(rel_pos.shape)[0], -1]).transpose([0, 2, 1]),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape([-1, max_rel_dist]).transpose([1, 0])
    else:
        rel_pos_resized = rel_pos

    q_coords = paddle.arange(end=q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = paddle.arange(end=k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = q_coords - k_coords + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.astype(dtype="int64")]


def add_decomposed_rel_pos(
    attn: paddle.Tensor,
    q: paddle.Tensor,
    rel_pos_h: paddle.Tensor,
    rel_pos_w: paddle.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> paddle.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = tuple(q.shape)
    r_q = q.reshape([B, q_h, q_w, dim])
    rel_h = paddle.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = paddle.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.reshape([B, q_h, q_w, k_h, k_w])
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).reshape([B, q_h * q_w, k_h * k_w])

    return attn


class PatchEmbed(paddle.nn.Layer):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.transpose([0, 2, 3, 1])
        return x


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"


class Qwen2LMHead(nn.Layer):
    def __init__(
        self,
        config,
        embedding_weights=None,
        transpose_y=False,
        tensor_parallel_output=1,
    ):
        super(Qwen2LMHead, self).__init__()
        self.config = config
        vocab_size = config.vocab_size

        self.transpose_y = transpose_y
        if transpose_y:
            # only for weight from embedding_weights
            if embedding_weights is not None:
                self.weight = embedding_weights
            else:
                self.weight = self.create_parameter(
                    shape=[vocab_size, config.hidden_size],
                    dtype=paddle.get_default_dtype(),
                )
        else:
            # for weight from model init
            self.weight = self.create_parameter(
                shape=[config.hidden_size, vocab_size],
                dtype=paddle.get_default_dtype(),
            )

    def forward(self, hidden_states, tensor_parallel_output=1):
        logits = paddle.matmul(hidden_states, self.weight, transpose_y=self.transpose_y)
        return logits


class GOTConfig(Qwen2Config):
    model_type = "GOT"


class GOTQwenModel(Qwen2Model):
    config_class = GOTConfig

    def __init__(self, config: Qwen2Config):
        super(GOTQwenModel, self).__init__(config)
        self.vision_tower_high = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(paddle.nn.LayerNorm, epsilon=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=256,
        )
        self.mm_projector_vary = nn.Linear(1024, 1024)

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        if orig_embeds_params is not None:
            with paddle.no_grad():
                self.get_input_embeddings().weight[: -self.num_new_tokens] = (
                    orig_embeds_params[: -self.num_new_tokens].data
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower_high = getattr(self, "vision_tower_high", None)

        if (
            vision_tower_high is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
        ):
            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)

            im_patch_token = 151859
            im_start_token = 151857
            im_end_token = 151858

            image_features = []

            for image in images:
                if self.training:
                    image = image[1]
                P, C, H, W = image.shape
                if P == 1:
                    with paddle.set_grad_enabled(False):
                        cnn_feature = vision_tower_high(image)
                        cnn_feature = cnn_feature.flatten(2).transpose(
                            [0, 2, 1]
                        )  # 256*1024
                    image_feature = self.mm_projector_vary(cnn_feature)
                    image_features.append(image_feature)

                else:
                    image_patches = paddle.unbind(image)
                    image_patches_features = []
                    for image_patch in image_patches:
                        image_p = paddle.stack([image_patch])
                        with paddle.set_grad_enabled(False):
                            cnn_feature_p = vision_tower_high(image_p)
                            cnn_feature_p = cnn_feature_p.flatten(2).transpose(
                                [0, 2, 1]
                            )
                        image_feature_p = self.mm_projector_vary(cnn_feature_p)
                        image_patches_features.append(image_feature_p)
                    image_feature = paddle.concat(image_patches_features, axis=1)
                    image_features.append(image_feature)

            dummy_image_features_2 = paddle.zeros(
                [256, 1024], dtype=inputs_embeds.dtype
            )
            dummy_image_features = dummy_image_features_2
            use_im_start_end = True
            new_input_embeds = []
            for cur_input_ids, cur_input_embeds, cur_image_features in zip(
                input_ids, inputs_embeds, image_features
            ):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (
                        cur_input_ids == im_end_token
                    ).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same."
                        )

                    image_start_tokens = paddle.where(cur_input_ids == im_start_token)[
                        0
                    ]
                    for image_start_token_pos, per_cur_image_features in zip(
                        image_start_tokens, cur_image_features
                    ):
                        num_patches = per_cur_image_features.shape[0]

                        if (
                            cur_input_ids[image_start_token_pos + num_patches + 1]
                            != im_end_token
                        ):
                            raise ValueError(
                                "The image end token should follow the image start token."
                            )

                        cur_input_embeds = paddle.concat(
                            (
                                cur_input_embeds[: image_start_token_pos + 1],
                                per_cur_image_features,
                                cur_input_embeds[
                                    image_start_token_pos + num_patches + 1 :
                                ],
                            ),
                            axis=0,
                        )

                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = paddle.stack(new_input_embeds, axis=0)

        return super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GOTQwenForCausalLM(Qwen2ForCausalLM):
    config_class = GOTConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.qwen2 = GOTQwenModel(config)

        self.vocab_size = config.vocab_size

        if config.tie_word_embeddings:
            self.lm_head = Qwen2LMHead(
                config,
                embedding_weights=self.qwen2.embed_tokens.weight,
                transpose_y=True,
            )
            self.tie_weights()
        else:
            self.lm_head = Qwen2LMHead(config)

        self.eval()

    def get_model(self):
        return self.qwen2

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.qwen2(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.astype(dtype="float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])

            loss = loss_fct(shift_logits, shift_labels)
            label_sum = paddle.sum(shift_labels != -100)
            loss = loss / label_sum

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
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
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        attention_mask = paddle.ones((batch_size, seq_length), dtype=paddle.bool)

        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[1]
            if past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype(dtype="int64").cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs


class PPChart2TableInference(GOTQwenForCausalLM):

    def generate(self, inputs, **kwargs):
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        no_repeat_ngram_size = kwargs.get("no_repeat_ngram_size", 20)

        with paddle.no_grad():
            generated_ids = super().generate(
                inputs["input_ids"],
                images=inputs["images"],
                do_sample=False,
                num_beams=1,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
            )

        return generated_ids
