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

# Code was based on https://github.com/Meituan-AutoML/Twins

from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.regularizer import L2Decay

from .vision_transformer import trunc_normal_, normal_, zeros_, ones_, to_2tuple, DropPath, Identity, Mlp
from .vision_transformer import Block as ViTBlock

from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "pcpvt_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_small_pretrained.pdparams",
    "pcpvt_base":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_base_pretrained.pdparams",
    "pcpvt_large":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/pcpvt_large_pretrained.pdparams",
    "alt_gvt_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_small_pretrained.pdparams",
    "alt_gvt_base":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_base_pretrained.pdparams",
    "alt_gvt_large":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/alt_gvt_large_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


class GroupAttention(nn.Layer):
    """LSA: self attention within a group.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 ws=1):
        super().__init__()
        if ws == 1:
            raise Exception("ws {ws} should not be 1")
        if dim % num_heads != 0:
            raise Exception(
                "dim {dim} should be divided by num_heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        x = x.reshape([B, h_group, self.ws, w_group, self.ws, C]).transpose(
            [0, 1, 3, 2, 4, 5])
        qkv = self.qkv(x).reshape([
            B, total_groups, self.ws**2, 3, self.num_heads, C // self.num_heads
        ]).transpose([3, 0, 1, 4, 2, 5])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = paddle.matmul(q, k.transpose([0, 1, 2, 4, 3])) * self.scale

        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)
        attn = paddle.matmul(attn, v).transpose([0, 1, 3, 2, 4]).reshape(
            [B, h_group, w_group, self.ws, self.ws, C])

        x = attn.transpose([0, 1, 3, 2, 4, 5]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Layer):
    """GSA: using a key to summarize the information for a group to be efficient.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2D(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(
            [B, N, self.num_heads, C // self.num_heads]).transpose(
                [0, 2, 1, 3])

        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1]).reshape([B, C, H, W])
            tmp_n = H * W // self.sr_ratio**2
            x_ = self.sr(x_).reshape([B, C, tmp_n]).transpose([0, 2, 1])
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(
                [B, tmp_n, 2, self.num_heads, C // self.num_heads]).transpose(
                    [2, 0, 3, 1, 4])
        else:
            kv = self.kv(x).reshape(
                [B, N, 2, self.num_heads, C // self.num_heads]).transpose(
                    [2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SBlock(ViTBlock):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer)

    def forward(self, x, H, W):
        return super().forward(x)


class GroupBlock(ViTBlock):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1,
                 ws=1):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop,
                         attn_drop, drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale,
                                  attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale,
                                       attn_drop, drop, ws)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if img_size % patch_size != 0:
            raise Exception(
                f"img_size {img_size} should be divided by patch_size {patch_size}."
            )

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


# borrow from PVT https://github.com/whai362/PVT.git
class PyramidVisionTransformer(nn.Layer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 block_cls=Block):
        super().__init__()
        self.class_num = class_num
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.LayerList()
        self.pos_embeds = nn.ParameterList()
        self.pos_drops = nn.LayerList()
        self.blocks = nn.LayerList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(
                    PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2**(i - 1), 2,
                               embed_dims[i - 1], embed_dims[i]))
            patch_num = self.patch_embeds[i].num_patches + 1 if i == len(
                embed_dims) - 1 else self.patch_embeds[i].num_patches
            self.pos_embeds.append(
                self.create_parameter(
                    shape=[1, patch_num, embed_dims[i]],
                    default_initializer=zeros_))
            self.pos_drops.append(nn.Dropout(p=drop_rate))

        dpr = [
            x.numpy()[0]
            for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        cur = 0
        for k in range(len(depths)):
            _block = nn.LayerList([
                block_cls(
                    dim=embed_dims[k],
                    num_heads=num_heads[k],
                    mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[k]) for i in range(depths[k])
            ])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = norm_layer(embed_dims[-1])

        # cls_token
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dims[-1]],
            default_initializer=zeros_,
            attr=paddle.ParamAttr(regularizer=L2Decay(0.0)))

        # classification head
        self.head = nn.Linear(embed_dims[-1],
                              class_num) if class_num > 0 else Identity()

        # init weights
        for pos_emb in self.pos_embeds:
            trunc_normal_(pos_emb)
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
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand([B, -1, -1])
                x = paddle.concat([cls_tokens, x], dim=1)
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = x.reshape([B, H, W, -1]).transpose(
                    [0, 3, 1, 2]).contiguous()
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Layer):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2D(
                in_chans,
                embed_dim,
                3,
                s,
                1,
                bias_attr=paddle.ParamAttr(regularizer=L2Decay(0.0)),
                groups=embed_dim,
                weight_attr=paddle.ParamAttr(regularizer=L2Decay(0.0)), ))
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose([0, 2, 1]).reshape([B, C, H, W])
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class CPVTV2(PyramidVisionTransformer):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 block_cls=Block):
        super().__init__(img_size, patch_size, in_chans, class_num, embed_dims,
                         num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                         attn_drop_rate, drop_path_rate, norm_layer, depths,
                         sr_ratios, block_cls)
        del self.pos_embeds
        del self.cls_token
        self.pos_block = nn.LayerList(
            [PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims])
        self.apply(self._init_weights)

    def _init_weights(self, m):
        import math
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            normal_(0, math.sqrt(2.0 / fan_out))(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)

            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here

            if i < len(self.depths) - 1:
                x = x.reshape([B, H, W, x.shape[-1]]).transpose([0, 3, 1, 2])

        x = self.norm(x)
        return x.mean(axis=1)  # GAP here


class PCPVT(CPVTV2):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1],
                 block_cls=SBlock):
        super().__init__(img_size, patch_size, in_chans, class_num, embed_dims,
                         num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                         attn_drop_rate, drop_path_rate, norm_layer, depths,
                         sr_ratios, block_cls)


class ALTGVT(PCPVT):
    """
    alias Twins-SVT
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4],
                 mlp_ratios=[4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4],
                 sr_ratios=[4, 2, 1],
                 block_cls=GroupBlock,
                 wss=[7, 7, 7]):
        super().__init__(img_size, patch_size, in_chans, class_num, embed_dims,
                         num_heads, mlp_ratios, qkv_bias, qk_scale, drop_rate,
                         attn_drop_rate, drop_path_rate, norm_layer, depths,
                         sr_ratios, block_cls)
        del self.blocks
        self.wss = wss
        # transformer encoder
        dpr = [
            x.numpy()[0]
            for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.LayerList()
        for k in range(len(depths)):
            _block = nn.LayerList([
                block_cls(
                    dim=embed_dims[k],
                    num_heads=num_heads[k],
                    mlp_ratio=mlp_ratios[k],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + i],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[k],
                    ws=1 if i % 2 == 1 else wss[k]) for i in range(depths[k])
            ])
            self.blocks.append(_block)
            cur += depths[k]
        self.apply(self._init_weights)


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


def pcpvt_small(pretrained=False, use_ssld=False, **kwargs):
    model = CPVTV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 4, 6, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["pcpvt_small"], use_ssld=use_ssld)
    return model


def pcpvt_base(pretrained=False, use_ssld=False, **kwargs):
    model = CPVTV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 4, 18, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["pcpvt_base"], use_ssld=use_ssld)
    return model


def pcpvt_large(pretrained=False, use_ssld=False, **kwargs):
    model = CPVTV2(
        patch_size=4,
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[3, 8, 27, 3],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["pcpvt_large"], use_ssld=use_ssld)
    return model


def alt_gvt_small(pretrained=False, use_ssld=False, **kwargs):
    model = ALTGVT(
        patch_size=4,
        embed_dims=[64, 128, 256, 512],
        num_heads=[2, 4, 8, 16],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 10, 4],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["alt_gvt_small"], use_ssld=use_ssld)
    return model


def alt_gvt_base(pretrained=False, use_ssld=False, **kwargs):
    model = ALTGVT(
        patch_size=4,
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 18, 2],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["alt_gvt_base"], use_ssld=use_ssld)
    return model


def alt_gvt_large(pretrained=False, use_ssld=False, **kwargs):
    model = ALTGVT(
        patch_size=4,
        embed_dims=[128, 256, 512, 1024],
        num_heads=[4, 8, 16, 32],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        depths=[2, 2, 18, 2],
        wss=[7, 7, 7, 7],
        sr_ratios=[8, 4, 2, 1],
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["alt_gvt_large"], use_ssld=use_ssld)
    return model
