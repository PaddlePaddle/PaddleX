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

# Code was based on https://github.com/ucbdrive/dla

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddle.nn.initializer import Normal, Constant

from paddlex.ppcls.arch.backbone.base.theseus_layer import Identity
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "DLA34":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA34_pretrained.pdparams",
    "DLA46_c":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA46_c_pretrained.pdparams",
    "DLA46x_c":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA46x_c_pretrained.pdparams",
    "DLA60":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60_pretrained.pdparams",
    "DLA60x":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_pretrained.pdparams",
    "DLA60x_c":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA60x_c_pretrained.pdparams",
    "DLA102":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102_pretrained.pdparams",
    "DLA102x":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x_pretrained.pdparams",
    "DLA102x2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA102x2_pretrained.pdparams",
    "DLA169":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DLA169_pretrained.pdparams"
}

__all__ = MODEL_URLS.keys()

zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class DlaBasic(nn.Layer):
    def __init__(self, inplanes, planes, stride=1, dilation=1, **cargs):
        super(DlaBasic, self).__init__()
        self.conv1 = nn.Conv2D(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias_attr=False,
            dilation=dilation)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias_attr=False,
            dilation=dilation)
        self.bn2 = nn.BatchNorm2D(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DlaBottleneck(nn.Layer):
    expansion = 2

    def __init__(self,
                 inplanes,
                 outplanes,
                 stride=1,
                 dilation=1,
                 cardinality=1,
                 base_width=64):
        super(DlaBottleneck, self).__init__()
        self.stride = stride
        mid_planes = int(
            math.floor(outplanes * (base_width / 64)) * cardinality)
        mid_planes = mid_planes // self.expansion

        self.conv1 = nn.Conv2D(
            inplanes, mid_planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(mid_planes)
        self.conv2 = nn.Conv2D(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias_attr=False,
            dilation=dilation,
            groups=cardinality)
        self.bn2 = nn.BatchNorm2D(mid_planes)
        self.conv3 = nn.Conv2D(
            mid_planes, outplanes, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(outplanes)
        self.relu = nn.ReLU()

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class DlaRoot(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(DlaRoot, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias_attr=False,
            padding=(kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(paddle.concat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class DlaTree(nn.Layer):
    def __init__(self,
                 levels,
                 block,
                 in_channels,
                 out_channels,
                 stride=1,
                 dilation=1,
                 cardinality=1,
                 base_width=64,
                 level_root=False,
                 root_dim=0,
                 root_kernel_size=1,
                 root_residual=False):
        super(DlaTree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        self.downsample = nn.MaxPool2D(
            stride, stride=stride) if stride > 1 else Identity()
        self.project = Identity()
        cargs = dict(
            dilation=dilation, cardinality=cardinality, base_width=base_width)

        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, **cargs)
            self.tree2 = block(out_channels, out_channels, 1, **cargs)
            if in_channels != out_channels:
                self.project = nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        bias_attr=False),
                    nn.BatchNorm2D(out_channels))
        else:
            cargs.update(
                dict(
                    root_kernel_size=root_kernel_size,
                    root_residual=root_residual))
            self.tree1 = DlaTree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                **cargs)
            self.tree2 = DlaTree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                **cargs)

        if levels == 1:
            self.root = DlaRoot(root_dim, out_channels, root_kernel_size,
                                root_residual)

        self.level_root = level_root
        self.root_dim = root_dim
        self.levels = levels

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)

        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)

        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Layer):
    def __init__(self,
                 levels,
                 channels,
                 in_chans=3,
                 cardinality=1,
                 base_width=64,
                 block=DlaBottleneck,
                 residual_root=False,
                 drop_rate=0.0,
                 class_num=1000,
                 with_pool=True):
        super(DLA, self).__init__()
        self.channels = channels
        self.class_num = class_num
        self.with_pool = with_pool
        self.cardinality = cardinality
        self.base_width = base_width
        self.drop_rate = drop_rate

        self.base_layer = nn.Sequential(
            nn.Conv2D(
                in_chans,
                channels[0],
                kernel_size=7,
                stride=1,
                padding=3,
                bias_attr=False),
            nn.BatchNorm2D(channels[0]),
            nn.ReLU())

        self.level0 = self._make_conv_level(channels[0], channels[0],
                                            levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)

        cargs = dict(
            cardinality=cardinality,
            base_width=base_width,
            root_residual=residual_root)

        self.level2 = DlaTree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            **cargs)
        self.level3 = DlaTree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            **cargs)
        self.level4 = DlaTree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            **cargs)
        self.level5 = DlaTree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            **cargs)

        self.feature_info = [
            # rare to have a meaningful stride 1 level
            dict(
                num_chs=channels[0], reduction=1, module='level0'),
            dict(
                num_chs=channels[1], reduction=2, module='level1'),
            dict(
                num_chs=channels[2], reduction=4, module='level2'),
            dict(
                num_chs=channels[3], reduction=8, module='level3'),
            dict(
                num_chs=channels[4], reduction=16, module='level4'),
            dict(
                num_chs=channels[5], reduction=32, module='level5'),
        ]

        self.num_features = channels[-1]

        if with_pool:
            self.global_pool = nn.AdaptiveAvgPool2D(1)

        if class_num > 0:
            self.fc = nn.Conv2D(self.num_features, class_num, 1)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
                normal_ = Normal(mean=0.0, std=math.sqrt(2. / n))
                normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                ones_(m.weight)
                zeros_(m.bias)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2D(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias_attr=False,
                    dilation=dilation), nn.BatchNorm2D(planes), nn.ReLU()
            ])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward_features(self, x):
        x = self.base_layer(x)

        x = self.level0(x)
        x = self.level1(x)
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.level5(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.with_pool:
            x = self.global_pool(x)

        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)

        if self.class_num > 0:
            x = self.fc(x)
            x = x.flatten(1)

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


def DLA34(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 2, 2, 1),
                channels=(16, 32, 64, 128, 256, 512),
                block=DlaBasic,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA34"])
    return model


def DLA46_c(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 2, 2, 1),
                channels=(16, 32, 64, 64, 128, 256),
                block=DlaBottleneck,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA46_c"])
    return model


def DLA46x_c(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 2, 2, 1),
                channels=(16, 32, 64, 64, 128, 256),
                block=DlaBottleneck,
                cardinality=32,
                base_width=4,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA46x_c"])
    return model


def DLA60(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 2, 3, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottleneck,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA60"])
    return model


def DLA60x(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 2, 3, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottleneck,
                cardinality=32,
                base_width=4,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA60x"])
    return model


def DLA60x_c(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 2, 3, 1),
                channels=(16, 32, 64, 64, 128, 256),
                block=DlaBottleneck,
                cardinality=32,
                base_width=4,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA60x_c"])
    return model


def DLA102(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 3, 4, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottleneck,
                residual_root=True,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA102"])
    return model


def DLA102x(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 3, 4, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottleneck,
                cardinality=32,
                base_width=4,
                residual_root=True,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA102x"])
    return model


def DLA102x2(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 1, 3, 4, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottleneck,
                cardinality=64,
                base_width=4,
                residual_root=True,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA102x2"])
    return model


def DLA169(pretrained=False, **kwargs):
    model = DLA(levels=(1, 1, 2, 3, 5, 1),
                channels=(16, 32, 128, 256, 512, 1024),
                block=DlaBottleneck,
                residual_root=True,
                **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["DLA169"])
    return model
