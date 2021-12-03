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

# Code was heavily based on https://github.com/facebookresearch/deit

import paddle
import paddle.nn as nn
from .vision_transformer import VisionTransformer, Identity, trunc_normal_, zeros_

from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "DeiT_tiny_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_patch16_224_pretrained.pdparams",
    "DeiT_small_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_patch16_224_pretrained.pdparams",
    "DeiT_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_224_pretrained.pdparams",
    "DeiT_tiny_distilled_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_tiny_distilled_patch16_224_pretrained.pdparams",
    "DeiT_small_distilled_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_small_distilled_patch16_224_pretrained.pdparams",
    "DeiT_base_distilled_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_224_pretrained.pdparams",
    "DeiT_base_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_patch16_384_pretrained.pdparams",
    "DeiT_base_distilled_patch16_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DeiT_base_distilled_patch16_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_num=class_num,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            epsilon=epsilon,
            **kwargs)
        self.pos_embed = self.create_parameter(
            shape=(1, self.patch_embed.num_patches + 2, self.embed_dim),
            default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)

        self.dist_token = self.create_parameter(
            shape=(1, 1, self.embed_dim), default_initializer=zeros_)
        self.add_parameter("cls_token", self.cls_token)

        self.head_dist = nn.Linear(
            self.embed_dim,
            self.class_num) if self.class_num > 0 else Identity()

        trunc_normal_(self.dist_token)
        trunc_normal_(self.pos_embed)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand((B, -1, -1))
        dist_token = self.dist_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, dist_token, x), axis=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return (x + x_dist) / 2


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


def DeiT_tiny_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_tiny_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_small_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_small_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_tiny_distilled_patch16_224(pretrained=False, use_ssld=False,
                                    **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_tiny_distilled_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_small_distilled_patch16_224(pretrained=False,
                                     use_ssld=False,
                                     **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_small_distilled_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_base_distilled_patch16_224(pretrained=False, use_ssld=False,
                                    **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_distilled_patch16_224"],
        use_ssld=use_ssld)
    return model


def DeiT_base_patch16_384(pretrained=False, use_ssld=False, **kwargs):
    model = VisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_patch16_384"],
        use_ssld=use_ssld)
    return model


def DeiT_base_distilled_patch16_384(pretrained=False, use_ssld=False,
                                    **kwargs):
    model = DistilledVisionTransformer(
        img_size=384,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["DeiT_base_distilled_patch16_384"],
        use_ssld=use_ssld)
    return model
