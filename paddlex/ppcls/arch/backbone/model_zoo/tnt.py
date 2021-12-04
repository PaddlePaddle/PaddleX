# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/huawei-noah/CV-Backbones/tree/master/tnt_pytorch

import math
import numpy as np

import paddle
import paddle.nn as nn

from paddle.nn.initializer import TruncatedNormal, Constant

from paddlex.ppcls.arch.backbone.base.theseus_layer import Identity
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "TNT_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/TNT_small_pretrained.pdparams"
}

__all__ = MODEL_URLS.keys()

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = paddle.add(keep_prob, paddle.rand(shape, dtype=x.dtype))
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 hidden_dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(
            (B, N, 2, self.num_heads, self.head_dim)).transpose(
                (2, 0, 3, 1, 4))

        q, k = qk[0], qk[1]
        v = self.v(x).reshape(
            (B, N, self.num_heads, x.shape[-1] // self.num_heads)).transpose(
                (0, 2, 1, 3))

        attn = paddle.matmul(q, k.transpose((0, 1, 3, 2))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v)
        x = x.transpose((0, 2, 1, 3)).reshape(
            (B, N, x.shape[-1] * x.shape[-3]))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 in_dim,
                 num_pixel,
                 num_heads=12,
                 in_num_head=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim,
            in_dim,
            num_heads=in_num_head,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim,
                          hidden_features=int(in_dim * 4),
                          out_features=in_dim,
                          act_layer=act_layer,
                          drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       out_features=dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, pixel_embed, patch_embed):
        # inner
        pixel_embed = paddle.add(
            pixel_embed,
            self.drop_path(self.attn_in(self.norm_in(pixel_embed))))
        pixel_embed = paddle.add(
            pixel_embed,
            self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed))))
        # outer
        B, N, C = patch_embed.shape
        norm1_proj = self.norm1_proj(pixel_embed)
        norm1_proj = norm1_proj.reshape(
            (B, N - 1, norm1_proj.shape[1] * norm1_proj.shape[2]))
        patch_embed[:, 1:] = paddle.add(patch_embed[:, 1:],
                                        self.proj(norm1_proj))
        patch_embed = paddle.add(
            patch_embed,
            self.drop_path(self.attn_out(self.norm_out(patch_embed))))
        patch_embed = paddle.add(
            patch_embed, self.drop_path(self.mlp(self.norm_mlp(patch_embed))))
        return pixel_embed, patch_embed


class PixelEmbed(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 in_dim=48,
                 stride=4):
        super().__init__()
        num_patches = (img_size // patch_size)**2
        self.img_size = img_size
        self.num_patches = num_patches
        self.in_dim = in_dim
        new_patch_size = math.ceil(patch_size / stride)
        self.new_patch_size = new_patch_size

        self.proj = nn.Conv2D(
            in_chans, self.in_dim, kernel_size=7, padding=3, stride=stride)

    def forward(self, x, pixel_pos):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        x = self.proj(x)
        x = nn.functional.unfold(x, self.new_patch_size, self.new_patch_size)
        x = x.transpose((0, 2, 1)).reshape(
            (-1, self.in_dim, self.new_patch_size, self.new_patch_size))
        x = x + pixel_pos
        x = x.reshape((-1, self.in_dim, self.new_patch_size *
                       self.new_patch_size)).transpose((0, 2, 1))
        return x


class TNT(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 in_dim=48,
                 depth=12,
                 num_heads=12,
                 in_num_head=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 first_stride=4,
                 class_num=1000):
        super().__init__()
        self.class_num = class_num
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim

        self.pixel_embed = PixelEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            in_dim=in_dim,
            stride=first_stride)
        num_patches = self.pixel_embed.num_patches
        self.num_patches = num_patches
        new_patch_size = self.pixel_embed.new_patch_size
        num_pixel = new_patch_size**2

        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        self.patch_pos = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=zeros_)
        self.add_parameter("patch_pos", self.patch_pos)

        self.pixel_pos = self.create_parameter(
            shape=(1, in_dim, new_patch_size, new_patch_size),
            default_initializer=zeros_)
        self.add_parameter("pixel_pos", self.pixel_pos)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, depth)

        blocks = []
        for i in range(depth):
            blocks.append(
                Block(
                    dim=embed_dim,
                    in_dim=in_dim,
                    num_pixel=num_pixel,
                    num_heads=num_heads,
                    in_num_head=in_num_head,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer))
        self.blocks = nn.LayerList(blocks)
        self.norm = norm_layer(embed_dim)

        if class_num > 0:
            self.head = nn.Linear(embed_dim, class_num)

        trunc_normal_(self.cls_token)
        trunc_normal_(self.patch_pos)
        trunc_normal_(self.pixel_pos)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        pixel_embed = self.pixel_embed(x, self.pixel_pos)

        patch_embed = self.norm2_proj(
            self.proj(
                self.norm1_proj(
                    pixel_embed.reshape((-1, self.num_patches, pixel_embed.
                                         shape[-1] * pixel_embed.shape[-2])))))
        patch_embed = paddle.concat(
            (self.cls_token.expand((B, -1, -1)), patch_embed), axis=1)
        patch_embed = patch_embed + self.patch_pos
        patch_embed = self.pos_drop(patch_embed)

        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed)

        patch_embed = self.norm(patch_embed)
        return patch_embed[:, 0]

    def forward(self, x):
        x = self.forward_features(x)

        if self.class_num > 0:
            x = self.head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def TNT_small(pretrained=False, **kwargs):
    model = TNT(patch_size=16,
                embed_dim=384,
                in_dim=24,
                depth=12,
                num_heads=6,
                in_num_head=4,
                qkv_bias=False,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["TNT_small"])
    return model
