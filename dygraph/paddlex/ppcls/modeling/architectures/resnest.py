# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.regularizer import L2Decay

__all__ = ["ResNeSt50_fast_1s1x64d", "ResNeSt50", "ResNeSt101"]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

        bn_decay = 0.0

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weight"),
            bias_attr=False)
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(
                name=name + "_scale", regularizer=L2Decay(bn_decay)),
            bias_attr=ParamAttr(
                name + "_offset", regularizer=L2Decay(bn_decay)),
            moving_mean_name=name + "_mean",
            moving_variance_name=name + "_variance")

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class rSoftmax(nn.Layer):
    def __init__(self, radix, cardinality):
        super(rSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        cardinality = self.cardinality
        radix = self.radix

        batch, r, h, w = x.shape
        if self.radix > 1:
            x = paddle.reshape(
                x=x,
                shape=[
                    batch, cardinality, radix,
                    int(r * h * w / cardinality / radix)
                ])
            x = paddle.transpose(x=x, perm=[0, 2, 1, 3])
            x = nn.functional.softmax(x, axis=1)
            x = paddle.reshape(x=x, shape=[batch, r * h * w, 1, 1])
        else:
            x = nn.functional.sigmoid(x)
        return x


class SplatConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 radix=2,
                 reduction_factor=4,
                 rectify_avg=False,
                 name=None):
        super(SplatConv, self).__init__()

        self.radix = radix

        self.conv1 = ConvBNLayer(
            num_channels=in_channels,
            num_filters=channels * radix,
            filter_size=kernel_size,
            stride=stride,
            groups=groups * radix,
            act="relu",
            name=name + "_1_weights")

        self.avg_pool2d = AdaptiveAvgPool2D(1)

        inter_channels = int(max(in_channels * radix // reduction_factor, 32))

        # to calc gap
        self.conv2 = ConvBNLayer(
            num_channels=channels,
            num_filters=inter_channels,
            filter_size=1,
            stride=1,
            groups=groups,
            act="relu",
            name=name + "_2_weights")

        # to calc atten
        self.conv3 = Conv2D(
            in_channels=inter_channels,
            out_channels=channels * radix,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            weight_attr=ParamAttr(
                name=name + "_weights", initializer=KaimingNormal()),
            bias_attr=False)

        self.rsoftmax = rSoftmax(radix=radix, cardinality=groups)

    def forward(self, x):
        x = self.conv1(x)

        if self.radix > 1:
            splited = paddle.split(x, num_or_sections=self.radix, axis=1)
            gap = paddle.add_n(splited)
        else:
            gap = x

        gap = self.avg_pool2d(gap)
        gap = self.conv2(gap)

        atten = self.conv3(gap)
        atten = self.rsoftmax(atten)

        if self.radix > 1:
            attens = paddle.split(atten, num_or_sections=self.radix, axis=1)
            y = paddle.add_n([
                paddle.multiply(split, att)
                for (att, split) in zip(attens, splited)
            ])
        else:
            y = paddle.multiply(x, atten)

        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 radix=1,
                 cardinality=1,
                 bottleneck_width=64,
                 avd=False,
                 avd_first=False,
                 dilation=1,
                 is_first=False,
                 rectify_avg=False,
                 last_gamma=False,
                 avg_down=False,
                 name=None):
        super(BottleneckBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.radix = radix
        self.cardinality = cardinality
        self.avd = avd
        self.avd_first = avd_first
        self.dilation = dilation
        self.is_first = is_first
        self.rectify_avg = rectify_avg
        self.last_gamma = last_gamma
        self.avg_down = avg_down

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality

        self.conv1 = ConvBNLayer(
            num_channels=self.inplanes,
            num_filters=group_width,
            filter_size=1,
            stride=1,
            groups=1,
            act="relu",
            name=name + "_conv1")

        if avd and avd_first and (stride > 1 or is_first):
            self.avg_pool2d_1 = AvgPool2D(
                kernel_size=3, stride=stride, padding=1)

        if radix >= 1:
            self.conv2 = SplatConv(
                in_channels=group_width,
                channels=group_width,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify_avg=rectify_avg,
                name=name + "_splat")
        else:
            self.conv2 = ConvBNLayer(
                num_channels=group_width,
                num_filters=group_width,
                filter_size=3,
                stride=1,
                dilation=dilation,
                groups=cardinality,
                act="relu",
                name=name + "_conv2")

        if avd and avd_first == False and (stride > 1 or is_first):
            self.avg_pool2d_2 = AvgPool2D(
                kernel_size=3, stride=stride, padding=1)

        self.conv3 = ConvBNLayer(
            num_channels=group_width,
            num_filters=planes * 4,
            filter_size=1,
            stride=1,
            groups=1,
            act=None,
            name=name + "_conv3")

        if stride != 1 or self.inplanes != self.planes * 4:
            if avg_down:
                if dilation == 1:
                    self.avg_pool2d_3 = AvgPool2D(
                        kernel_size=stride, stride=stride, padding=0)
                else:
                    self.avg_pool2d_3 = AvgPool2D(
                        kernel_size=1, stride=1, padding=0, ceil_mode=True)

                self.conv4 = Conv2D(
                    in_channels=self.inplanes,
                    out_channels=planes * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    weight_attr=ParamAttr(
                        name=name + "_weights", initializer=KaimingNormal()),
                    bias_attr=False)
            else:
                self.conv4 = Conv2D(
                    in_channels=self.inplanes,
                    out_channels=planes * 4,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=1,
                    weight_attr=ParamAttr(
                        name=name + "_shortcut_weights",
                        initializer=KaimingNormal()),
                    bias_attr=False)

            bn_decay = 0.0
            self._batch_norm = BatchNorm(
                planes * 4,
                act=None,
                param_attr=ParamAttr(
                    name=name + "_shortcut_scale",
                    regularizer=L2Decay(bn_decay)),
                bias_attr=ParamAttr(
                    name + "_shortcut_offset", regularizer=L2Decay(bn_decay)),
                moving_mean_name=name + "_shortcut_mean",
                moving_variance_name=name + "_shortcut_variance")

    def forward(self, x):
        short = x

        x = self.conv1(x)
        if self.avd and self.avd_first and (self.stride > 1 or self.is_first):
            x = self.avg_pool2d_1(x)

        x = self.conv2(x)

        if self.avd and self.avd_first == False and (self.stride > 1 or
                                                     self.is_first):
            x = self.avg_pool2d_2(x)

        x = self.conv3(x)

        if self.stride != 1 or self.inplanes != self.planes * 4:
            if self.avg_down:
                short = self.avg_pool2d_3(short)

            short = self.conv4(short)

            short = self._batch_norm(short)

        y = paddle.add(x=short, y=x)
        y = F.relu(y)
        return y


class ResNeStLayer(nn.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 blocks,
                 radix,
                 cardinality,
                 bottleneck_width,
                 avg_down,
                 avd,
                 avd_first,
                 rectify_avg,
                 last_gamma,
                 stride=1,
                 dilation=1,
                 is_first=True,
                 name=None):
        super(ResNeStLayer, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.blocks = blocks
        self.radix = radix
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.avg_down = avg_down
        self.avd = avd
        self.avd_first = avd_first
        self.rectify_avg = rectify_avg
        self.last_gamma = last_gamma
        self.is_first = is_first

        if dilation == 1 or dilation == 2:
            bottleneck_func = self.add_sublayer(
                name + "_bottleneck_0",
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=1,
                    is_first=is_first,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    name=name + "_bottleneck_0"))
        elif dilation == 4:
            bottleneck_func = self.add_sublayer(
                name + "_bottleneck_0",
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    stride=stride,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=2,
                    is_first=is_first,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    name=name + "_bottleneck_0"))
        else:
            raise RuntimeError("=>unknown dilation size")

        self.inplanes = planes * 4
        self.bottleneck_block_list = [bottleneck_func]
        for i in range(1, blocks):
            curr_name = name + "_bottleneck_" + str(i)

            bottleneck_func = self.add_sublayer(
                curr_name,
                BottleneckBlock(
                    inplanes=self.inplanes,
                    planes=planes,
                    radix=radix,
                    cardinality=cardinality,
                    bottleneck_width=bottleneck_width,
                    avg_down=self.avg_down,
                    avd=avd,
                    avd_first=avd_first,
                    dilation=dilation,
                    rectify_avg=rectify_avg,
                    last_gamma=last_gamma,
                    name=curr_name))
            self.bottleneck_block_list.append(bottleneck_func)

    def forward(self, x):
        for bottleneck_block in self.bottleneck_block_list:
            x = bottleneck_block(x)
        return x


class ResNeSt(nn.Layer):
    def __init__(self,
                 layers,
                 radix=1,
                 groups=1,
                 bottleneck_width=64,
                 dilated=False,
                 dilation=1,
                 deep_stem=False,
                 stem_width=64,
                 avg_down=False,
                 rectify_avg=False,
                 avd=False,
                 avd_first=False,
                 final_drop=0.0,
                 last_gamma=False,
                 class_dim=1000):
        super(ResNeSt, self).__init__()

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.layers = layers
        self.final_drop = final_drop
        self.dilated = dilated
        self.dilation = dilation

        self.rectify_avg = rectify_avg

        if self.deep_stem:
            self.stem = nn.Sequential(
                ("conv1", ConvBNLayer(
                    num_channels=3,
                    num_filters=stem_width,
                    filter_size=3,
                    stride=2,
                    act="relu",
                    name="conv1")), ("conv2", ConvBNLayer(
                        num_channels=stem_width,
                        num_filters=stem_width,
                        filter_size=3,
                        stride=1,
                        act="relu",
                        name="conv2")), ("conv3", ConvBNLayer(
                            num_channels=stem_width,
                            num_filters=stem_width * 2,
                            filter_size=3,
                            stride=1,
                            act="relu",
                            name="conv3")))
        else:
            self.stem = ConvBNLayer(
                num_channels=3,
                num_filters=stem_width,
                filter_size=7,
                stride=2,
                act="relu",
                name="conv1")

        self.max_pool2d = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResNeStLayer(
            inplanes=self.stem_width * 2
            if self.deep_stem else self.stem_width,
            planes=64,
            blocks=self.layers[0],
            radix=radix,
            cardinality=self.cardinality,
            bottleneck_width=bottleneck_width,
            avg_down=self.avg_down,
            avd=avd,
            avd_first=avd_first,
            rectify_avg=rectify_avg,
            last_gamma=last_gamma,
            stride=1,
            dilation=1,
            is_first=False,
            name="layer1")

        #         return

        self.layer2 = ResNeStLayer(
            inplanes=256,
            planes=128,
            blocks=self.layers[1],
            radix=radix,
            cardinality=self.cardinality,
            bottleneck_width=bottleneck_width,
            avg_down=self.avg_down,
            avd=avd,
            avd_first=avd_first,
            rectify_avg=rectify_avg,
            last_gamma=last_gamma,
            stride=2,
            name="layer2")

        if self.dilated or self.dilation == 4:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=2,
                name="layer3")
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=4,
                name="layer4")
        elif self.dilation == 2:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                dilation=1,
                name="layer3")
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=1,
                dilation=2,
                name="layer4")
        else:
            self.layer3 = ResNeStLayer(
                inplanes=512,
                planes=256,
                blocks=self.layers[2],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                name="layer3")
            self.layer4 = ResNeStLayer(
                inplanes=1024,
                planes=512,
                blocks=self.layers[3],
                radix=radix,
                cardinality=self.cardinality,
                bottleneck_width=bottleneck_width,
                avg_down=self.avg_down,
                avd=avd,
                avd_first=avd_first,
                rectify_avg=rectify_avg,
                last_gamma=last_gamma,
                stride=2,
                name="layer4")

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.out_channels = 2048

        stdv = 1.0 / math.sqrt(self.out_channels * 1.0)

        self.out = Linear(
            self.out_channels,
            class_dim,
            weight_attr=ParamAttr(
                initializer=nn.initializer.Uniform(-stdv, stdv),
                name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, x):
        x = self.stem(x)
        x = self.max_pool2d(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)
        x = self.pool2d_avg(x)
        x = paddle.reshape(x, shape=[-1, self.out_channels])
        x = self.out(x)
        return x


def ResNeSt50_fast_1s1x64d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=1,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.0,
        **args)
    return model


def ResNeSt101(**args):
    model = ResNeSt(
        layers=[3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.0,
        **args)
    return model
