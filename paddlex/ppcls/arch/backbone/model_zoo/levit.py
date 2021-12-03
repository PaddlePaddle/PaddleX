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

# Code was based on https://github.com/facebookresearch/LeViT

import itertools
import math
import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant
from paddle.regularizer import L2Decay

from .vision_transformer import trunc_normal_, zeros_, ones_, Identity

from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "LeViT_128S":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128S_pretrained.pdparams",
    "LeViT_128":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_128_pretrained.pdparams",
    "LeViT_192":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_192_pretrained.pdparams",
    "LeViT_256":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_256_pretrained.pdparams",
    "LeViT_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/LeViT_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


def cal_attention_biases(attention_biases, attention_bias_idxs):
    gather_list = []
    attention_bias_t = paddle.transpose(attention_biases, (1, 0))
    nums = attention_bias_idxs.shape[0]
    for idx in range(nums):
        gather = paddle.gather(attention_bias_t, attention_bias_idxs[idx])
        gather_list.append(gather)
    shape0, shape1 = attention_bias_idxs.shape
    gather = paddle.concat(gather_list)
    return paddle.transpose(gather, (1, 0)).reshape((0, shape0, shape1))


class Conv2d_BN(nn.Sequential):
    def __init__(self,
                 a,
                 b,
                 ks=1,
                 stride=1,
                 pad=0,
                 dilation=1,
                 groups=1,
                 bn_weight_init=1,
                 resolution=-10000):
        super().__init__()
        self.add_sublayer(
            'c',
            nn.Conv2D(
                a, b, ks, stride, pad, dilation, groups, bias_attr=False))
        bn = nn.BatchNorm2D(b)
        ones_(bn.weight)
        zeros_(bn.bias)
        self.add_sublayer('bn', bn)


class Linear_BN(nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_sublayer('c', nn.Linear(a, b, bias_attr=False))
        bn = nn.BatchNorm1D(b)
        if bn_weight_init == 0:
            zeros_(bn.weight)
        else:
            ones_(bn.weight)
        zeros_(bn.bias)
        self.add_sublayer('bn', bn)

    def forward(self, x):
        l, bn = self._sub_layers.values()
        x = l(x)
        return paddle.reshape(bn(x.flatten(0, 1)), x.shape)


class BN_Linear(nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_sublayer('bn', nn.BatchNorm1D(a))
        l = nn.Linear(a, b, bias_attr=bias)
        trunc_normal_(l.weight)
        if bias:
            zeros_(l.bias)
        self.add_sublayer('l', l)


def b16(n, activation, resolution=224):
    return nn.Sequential(
        Conv2d_BN(
            3, n // 8, 3, 2, 1, resolution=resolution),
        activation(),
        Conv2d_BN(
            n // 8, n // 4, 3, 2, 1, resolution=resolution // 2),
        activation(),
        Conv2d_BN(
            n // 4, n // 2, 3, 2, 1, resolution=resolution // 4),
        activation(),
        Conv2d_BN(
            n // 2, n, 3, 2, 1, resolution=resolution // 8))


class Residual(nn.Layer):
    def __init__(self, m, drop):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            y = paddle.rand(
                shape=[x.shape[0], 1, 1]).__ge__(self.drop).astype("float32")
            y = y.divide(paddle.full_like(y, 1 - self.drop))
            return paddle.add(x, y)
        else:
            return paddle.add(x, self.m(x))


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=4,
                 activation=None,
                 resolution=14):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.h = self.dh + nh_kd * 2
        self.qkv = Linear_BN(dim, self.h)
        self.proj = nn.Sequential(
            activation(), Linear_BN(
                self.dh, dim, bn_weight_init=0))
        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(
            shape=(num_heads, len(attention_offsets)),
            default_initializer=zeros_,
            attr=paddle.ParamAttr(regularizer=L2Decay(0.0)))
        tensor_idxs = paddle.to_tensor(idxs, dtype='int64')
        self.register_buffer('attention_bias_idxs',
                             paddle.reshape(tensor_idxs, [N, N]))

    @paddle.no_grad()
    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = cal_attention_biases(self.attention_biases,
                                           self.attention_bias_idxs)

    def forward(self, x):
        self.training = True
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = paddle.reshape(qkv,
                             [B, N, self.num_heads, self.h // self.num_heads])
        q, k, v = paddle.split(
            qkv, [self.key_dim, self.key_dim, self.d], axis=3)
        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        k_transpose = paddle.transpose(k, perm=[0, 1, 3, 2])

        if self.training:
            attention_biases = cal_attention_biases(self.attention_biases,
                                                    self.attention_bias_idxs)
        else:
            attention_biases = self.ab
        attn = (paddle.matmul(q, k_transpose) * self.scale + attention_biases)
        attn = F.softmax(attn)
        x = paddle.transpose(paddle.matmul(attn, v), perm=[0, 2, 1, 3])
        x = paddle.reshape(x, [B, N, self.dh])
        x = self.proj(x)
        return x


class Subsample(nn.Layer):
    def __init__(self, stride, resolution):
        super().__init__()
        self.stride = stride
        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        x = paddle.reshape(x, [B, self.resolution, self.resolution, C])
        end1, end2 = x.shape[1], x.shape[2]
        x = x[:, 0:end1:self.stride, 0:end2:self.stride]
        x = paddle.reshape(x, [B, -1, C])
        return x


class AttentionSubsample(nn.Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=2,
                 activation=None,
                 stride=2,
                 resolution=14,
                 resolution_=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        self.resolution_ = resolution_
        self.resolution_2 = resolution_**2
        self.training = True
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h)

        self.q = nn.Sequential(
            Subsample(stride, resolution), Linear_BN(in_dim, nh_kd))
        self.proj = nn.Sequential(activation(), Linear_BN(self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution
        points = list(itertools.product(range(resolution), range(resolution)))
        points_ = list(
            itertools.product(range(resolution_), range(resolution_)))

        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        i = 0
        j = 0
        for p1 in points_:
            i += 1
            for p2 in points:
                j += 1
                size = 1
                offset = (abs(p1[0] * stride - p2[0] + (size - 1) / 2),
                          abs(p1[1] * stride - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(
            shape=(num_heads, len(attention_offsets)),
            default_initializer=zeros_,
            attr=paddle.ParamAttr(regularizer=L2Decay(0.0)))

        tensor_idxs_ = paddle.to_tensor(idxs, dtype='int64')
        self.register_buffer('attention_bias_idxs',
                             paddle.reshape(tensor_idxs_, [N_, N]))

    @paddle.no_grad()
    def train(self, mode=True):
        if mode:
            super().train()
        else:
            super().eval()
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = cal_attention_biases(self.attention_biases,
                                           self.attention_bias_idxs)

    def forward(self, x):
        self.training = True
        B, N, C = x.shape
        kv = self.kv(x)
        kv = paddle.reshape(kv, [B, N, self.num_heads, -1])
        k, v = paddle.split(kv, [self.key_dim, self.d], axis=3)
        k = paddle.transpose(k, perm=[0, 2, 1, 3])  # BHNC
        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        q = paddle.reshape(
            self.q(x), [B, self.resolution_2, self.num_heads, self.key_dim])
        q = paddle.transpose(q, perm=[0, 2, 1, 3])

        if self.training:
            attention_biases = cal_attention_biases(self.attention_biases,
                                                    self.attention_bias_idxs)
        else:
            attention_biases = self.ab

        attn = (paddle.matmul(
            q, paddle.transpose(
                k, perm=[0, 1, 3, 2]))) * self.scale + attention_biases
        attn = F.softmax(attn)

        x = paddle.reshape(
            paddle.transpose(
                paddle.matmul(attn, v), perm=[0, 2, 1, 3]), [B, -1, self.dh])
        x = self.proj(x)
        return x


class LeViT(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=[192],
                 key_dim=[64],
                 depth=[12],
                 num_heads=[3],
                 attn_ratio=[2],
                 mlp_ratio=[2],
                 hybrid_backbone=None,
                 down_ops=[],
                 attention_activation=nn.Hardswish,
                 mlp_activation=nn.Hardswish,
                 distillation=True,
                 drop_path=0):
        super().__init__()

        self.class_num = class_num
        self.num_features = embed_dim[-1]
        self.embed_dim = embed_dim
        self.distillation = distillation

        self.patch_embed = hybrid_backbone

        self.blocks = []
        down_ops.append([''])
        resolution = img_size // patch_size
        for i, (ed, kd, dpth, nh, ar, mr, do) in enumerate(
                zip(embed_dim, key_dim, depth, num_heads, attn_ratio,
                    mlp_ratio, down_ops)):
            for _ in range(dpth):
                self.blocks.append(
                    Residual(
                        Attention(
                            ed,
                            kd,
                            nh,
                            attn_ratio=ar,
                            activation=attention_activation,
                            resolution=resolution, ),
                        drop_path))
                if mr > 0:
                    h = int(ed * mr)
                    self.blocks.append(
                        Residual(
                            nn.Sequential(
                                Linear_BN(ed, h),
                                mlp_activation(),
                                Linear_BN(
                                    h, ed, bn_weight_init=0), ),
                            drop_path))
            if do[0] == 'Subsample':
                #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
                resolution_ = (resolution - 1) // do[5] + 1
                self.blocks.append(
                    AttentionSubsample(
                        *embed_dim[i:i + 2],
                        key_dim=do[1],
                        num_heads=do[2],
                        attn_ratio=do[3],
                        activation=attention_activation,
                        stride=do[5],
                        resolution=resolution,
                        resolution_=resolution_))
                resolution = resolution_
                if do[4] > 0:  # mlp_ratio
                    h = int(embed_dim[i + 1] * do[4])
                    self.blocks.append(
                        Residual(
                            nn.Sequential(
                                Linear_BN(embed_dim[i + 1], h),
                                mlp_activation(),
                                Linear_BN(
                                    h, embed_dim[i + 1], bn_weight_init=0), ),
                            drop_path))
        self.blocks = nn.Sequential(*self.blocks)

        # Classifier head
        self.head = BN_Linear(embed_dim[-1],
                              class_num) if class_num > 0 else Identity()
        if distillation:
            self.head_dist = BN_Linear(
                embed_dim[-1], class_num) if class_num > 0 else Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2)
        x = paddle.transpose(x, perm=[0, 2, 1])
        x = self.blocks(x)
        x = x.mean(1)

        x = paddle.reshape(x, [-1, self.embed_dim[-1]])
        if self.distillation:
            x = self.head(x), self.head_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.head(x)
        return x


def model_factory(C, D, X, N, drop_path, class_num, distillation):
    embed_dim = [int(x) for x in C.split('_')]
    num_heads = [int(x) for x in N.split('_')]
    depth = [int(x) for x in X.split('_')]
    act = nn.Hardswish
    model = LeViT(
        patch_size=16,
        embed_dim=embed_dim,
        num_heads=num_heads,
        key_dim=[D] * 3,
        depth=depth,
        attn_ratio=[2, 2, 2],
        mlp_ratio=[2, 2, 2],
        down_ops=[
            #('Subsample',key_dim, num_heads, attn_ratio, mlp_ratio, stride)
            ['Subsample', D, embed_dim[0] // D, 4, 2, 2],
            ['Subsample', D, embed_dim[1] // D, 4, 2, 2],
        ],
        attention_activation=act,
        mlp_activation=act,
        hybrid_backbone=b16(embed_dim[0], activation=act),
        class_num=class_num,
        drop_path=drop_path,
        distillation=distillation)

    return model


specification = {
    'LeViT_128S': {
        'C': '128_256_384',
        'D': 16,
        'N': '4_6_8',
        'X': '2_3_4',
        'drop_path': 0
    },
    'LeViT_128': {
        'C': '128_256_384',
        'D': 16,
        'N': '4_8_12',
        'X': '4_4_4',
        'drop_path': 0
    },
    'LeViT_192': {
        'C': '192_288_384',
        'D': 32,
        'N': '3_5_6',
        'X': '4_4_4',
        'drop_path': 0
    },
    'LeViT_256': {
        'C': '256_384_512',
        'D': 32,
        'N': '4_6_8',
        'X': '4_4_4',
        'drop_path': 0
    },
    'LeViT_384': {
        'C': '384_512_768',
        'D': 32,
        'N': '6_9_12',
        'X': '4_4_4',
        'drop_path': 0.1
    },
}


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


def LeViT_128S(pretrained=False,
               use_ssld=False,
               class_num=1000,
               distillation=False,
               **kwargs):
    model = model_factory(
        **specification['LeViT_128S'],
        class_num=class_num,
        distillation=distillation)
    _load_pretrained(
        pretrained, model, MODEL_URLS["LeViT_128S"], use_ssld=use_ssld)
    return model


def LeViT_128(pretrained=False,
              use_ssld=False,
              class_num=1000,
              distillation=False,
              **kwargs):
    model = model_factory(
        **specification['LeViT_128'],
        class_num=class_num,
        distillation=distillation)
    _load_pretrained(
        pretrained, model, MODEL_URLS["LeViT_128"], use_ssld=use_ssld)
    return model


def LeViT_192(pretrained=False,
              use_ssld=False,
              class_num=1000,
              distillation=False,
              **kwargs):
    model = model_factory(
        **specification['LeViT_192'],
        class_num=class_num,
        distillation=distillation)
    _load_pretrained(
        pretrained, model, MODEL_URLS["LeViT_192"], use_ssld=use_ssld)
    return model


def LeViT_256(pretrained=False,
              use_ssld=False,
              class_num=1000,
              distillation=False,
              **kwargs):
    model = model_factory(
        **specification['LeViT_256'],
        class_num=class_num,
        distillation=distillation)
    _load_pretrained(
        pretrained, model, MODEL_URLS["LeViT_256"], use_ssld=use_ssld)
    return model


def LeViT_384(pretrained=False,
              use_ssld=False,
              class_num=1000,
              distillation=False,
              **kwargs):
    model = model_factory(
        **specification['LeViT_384'],
        class_num=class_num,
        distillation=distillation)
    _load_pretrained(
        pretrained, model, MODEL_URLS["LeViT_384"], use_ssld=use_ssld)
    return model
