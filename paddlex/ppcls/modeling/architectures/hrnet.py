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
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

__all__ = [
    "HRNet_W18_C",
    "HRNet_W30_C",
    "HRNet_W32_C",
    "HRNet_W40_C",
    "HRNet_W44_C",
    "HRNet_W48_C",
    "HRNet_W60_C",
    "HRNet_W64_C",
    "SE_HRNet_W18_C",
    "SE_HRNet_W30_C",
    "SE_HRNet_W32_C",
    "SE_HRNet_W40_C",
    "SE_HRNet_W44_C",
    "SE_HRNet_W48_C",
    "SE_HRNet_W60_C",
    "SE_HRNet_W64_C",
]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act="relu",
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = name + '_bn'
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, input):
        y = self._conv(input)
        y = self._batch_norm(y)
        return y


class Layer1(nn.Layer):
    def __init__(self, num_channels, has_se=False, name=None):
        super(Layer1, self).__init__()

        self.bottleneck_block_list = []

        for i in range(4):
            bottleneck_block = self.add_sublayer(
                "bb_{}_{}".format(name, i + 1),
                BottleneckBlock(
                    num_channels=num_channels if i == 0 else 256,
                    num_filters=64,
                    has_se=has_se,
                    stride=1,
                    downsample=True if i == 0 else False,
                    name=name + '_' + str(i + 1)))
            self.bottleneck_block_list.append(bottleneck_block)

    def forward(self, input):
        conv = input
        for block_func in self.bottleneck_block_list:
            conv = block_func(conv)
        return conv


class TransitionLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, name=None):
        super(TransitionLayer, self).__init__()

        num_in = len(in_channels)
        num_out = len(out_channels)
        out = []
        self.conv_bn_func_list = []
        for i in range(num_out):
            residual = None
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.add_sublayer(
                        "transition_{}_layer_{}".format(name, i + 1),
                        ConvBNLayer(
                            num_channels=in_channels[i],
                            num_filters=out_channels[i],
                            filter_size=3,
                            name=name + '_layer_' + str(i + 1)))
            else:
                residual = self.add_sublayer(
                    "transition_{}_layer_{}".format(name, i + 1),
                    ConvBNLayer(
                        num_channels=in_channels[-1],
                        num_filters=out_channels[i],
                        filter_size=3,
                        stride=2,
                        name=name + '_layer_' + str(i + 1)))
            self.conv_bn_func_list.append(residual)

    def forward(self, input):
        outs = []
        for idx, conv_bn_func in enumerate(self.conv_bn_func_list):
            if conv_bn_func is None:
                outs.append(input[idx])
            else:
                if idx < len(input):
                    outs.append(conv_bn_func(input[idx]))
                else:
                    outs.append(conv_bn_func(input[-1]))
        return outs


class Branches(nn.Layer):
    def __init__(self,
                 block_num,
                 in_channels,
                 out_channels,
                 has_se=False,
                 name=None):
        super(Branches, self).__init__()

        self.basic_block_list = []

        for i in range(len(out_channels)):
            self.basic_block_list.append([])
            for j in range(block_num):
                in_ch = in_channels[i] if j == 0 else out_channels[i]
                basic_block_func = self.add_sublayer(
                    "bb_{}_branch_layer_{}_{}".format(name, i + 1, j + 1),
                    BasicBlock(
                        num_channels=in_ch,
                        num_filters=out_channels[i],
                        has_se=has_se,
                        name=name + '_branch_layer_' + str(i + 1) + '_' +
                        str(j + 1)))
                self.basic_block_list[i].append(basic_block_func)

    def forward(self, inputs):
        outs = []
        for idx, input in enumerate(inputs):
            conv = input
            basic_block_list = self.basic_block_list[idx]
            for basic_block_func in basic_block_list:
                conv = basic_block_func(conv)
            outs.append(conv)
        return outs


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu",
            name=name + "_conv1", )
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            name=name + "_conv2")
        self.conv3 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_conv3")

        if self.downsample:
            self.conv_down = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                act=None,
                name=name + "_downsample")

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16,
                name='fc' + name)

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(input)

        if self.has_se:
            conv3 = self.se(conv3)

        y = paddle.add(x=residual, y=conv3)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 has_se=False,
                 downsample=False,
                 name=None):
        super(BasicBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu",
            name=name + "_conv1")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act=None,
            name=name + "_conv2")

        if self.downsample:
            self.conv_down = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                act="relu",
                name=name + "_downsample")

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16,
                name='fc' + name)

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)

        if self.downsample:
            residual = self.conv_down(input)

        if self.has_se:
            conv2 = self.se(conv2)

        y = paddle.add(x=residual, y=conv2)
        y = F.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_sqz_weights"),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_filters,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name=name + "_exc_weights"),
            bias_attr=ParamAttr(name=name + '_exc_offset'))

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = input * excitation
        return out


class Stage(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_modules,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = []
        for i in range(num_modules):
            if i == num_modules - 1 and not multi_scale_output:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_filters=num_filters,
                        has_se=has_se,
                        multi_scale_output=False,
                        name=name + '_' + str(i + 1)))
            else:
                stage_func = self.add_sublayer(
                    "stage_{}_{}".format(name, i + 1),
                    HighResolutionModule(
                        num_channels=num_channels,
                        num_filters=num_filters,
                        has_se=has_se,
                        name=name + '_' + str(i + 1)))

            self.stage_func_list.append(stage_func)

    def forward(self, input):
        out = input
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se=False,
                 multi_scale_output=True,
                 name=None):
        super(HighResolutionModule, self).__init__()

        self.branches_func = Branches(
            block_num=4,
            in_channels=num_channels,
            out_channels=num_filters,
            has_se=has_se,
            name=name)

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters,
            multi_scale_output=multi_scale_output,
            name=name)

    def forward(self, input):
        out = self.branches_func(input)
        out = self.fuse_func(out)
        return out


class FuseLayers(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 multi_scale_output=True,
                 name=None):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels) if multi_scale_output else 1
        self._in_channels = in_channels

        self.residual_func_list = []
        for i in range(self._actual_ch):
            for j in range(len(in_channels)):
                residual_func = None
                if j > i:
                    residual_func = self.add_sublayer(
                        "residual_{}_layer_{}_{}".format(name, i + 1, j + 1),
                        ConvBNLayer(
                            num_channels=in_channels[j],
                            num_filters=out_channels[i],
                            filter_size=1,
                            stride=1,
                            act=None,
                            name=name + '_layer_' + str(i + 1) + '_' +
                            str(j + 1)))
                    self.residual_func_list.append(residual_func)
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[i],
                                    filter_size=3,
                                    stride=2,
                                    act=None,
                                    name=name + '_layer_' + str(i + 1) + '_' +
                                    str(j + 1) + '_' + str(k + 1)))
                            pre_num_filters = out_channels[i]
                        else:
                            residual_func = self.add_sublayer(
                                "residual_{}_layer_{}_{}_{}".format(
                                    name, i + 1, j + 1, k + 1),
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[j],
                                    filter_size=3,
                                    stride=2,
                                    act="relu",
                                    name=name + '_layer_' + str(i + 1) + '_' +
                                    str(j + 1) + '_' + str(k + 1)))
                            pre_num_filters = out_channels[j]
                        self.residual_func_list.append(residual_func)

    def forward(self, input):
        outs = []
        residual_func_idx = 0
        for i in range(self._actual_ch):
            residual = input[i]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](input[j])
                    residual_func_idx += 1

                    y = F.upsample(y, scale_factor=2**(j - i), mode="nearest")
                    residual = paddle.add(x=residual, y=y)
                elif j < i:
                    y = input[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = paddle.add(x=residual, y=y)

            residual = F.relu(residual)
            outs.append(residual)

        return outs


class LastClsOut(nn.Layer):
    def __init__(self,
                 num_channel_list,
                 has_se,
                 num_filters_list=[32, 64, 128, 256],
                 name=None):
        super(LastClsOut, self).__init__()

        self.func_list = []
        for idx in range(len(num_channel_list)):
            func = self.add_sublayer(
                "conv_{}_conv_{}".format(name, idx + 1),
                BottleneckBlock(
                    num_channels=num_channel_list[idx],
                    num_filters=num_filters_list[idx],
                    has_se=has_se,
                    downsample=True,
                    name=name + 'conv_' + str(idx + 1)))
            self.func_list.append(func)

    def forward(self, inputs):
        outs = []
        for idx, input in enumerate(inputs):
            out = self.func_list[idx](input)
            outs.append(out)
        return outs


class HRNet(nn.Layer):
    def __init__(self, width=18, has_se=False, class_dim=1000):
        super(HRNet, self).__init__()

        self.width = width
        self.has_se = has_se
        self.channels = {
            18: [[18, 36], [18, 36, 72], [18, 36, 72, 144]],
            30: [[30, 60], [30, 60, 120], [30, 60, 120, 240]],
            32: [[32, 64], [32, 64, 128], [32, 64, 128, 256]],
            40: [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            44: [[44, 88], [44, 88, 176], [44, 88, 176, 352]],
            48: [[48, 96], [48, 96, 192], [48, 96, 192, 384]],
            60: [[60, 120], [60, 120, 240], [60, 120, 240, 480]],
            64: [[64, 128], [64, 128, 256], [64, 128, 256, 512]]
        }
        self._class_dim = class_dim

        channels_2, channels_3, channels_4 = self.channels[width]
        num_modules_2, num_modules_3, num_modules_4 = 1, 4, 3

        self.conv_layer1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu',
            name="layer1_1")

        self.conv_layer1_2 = ConvBNLayer(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu',
            name="layer1_2")

        self.la1 = Layer1(num_channels=64, has_se=has_se, name="layer2")

        self.tr1 = TransitionLayer(
            in_channels=[256], out_channels=channels_2, name="tr1")

        self.st2 = Stage(
            num_channels=channels_2,
            num_modules=num_modules_2,
            num_filters=channels_2,
            has_se=self.has_se,
            name="st2")

        self.tr2 = TransitionLayer(
            in_channels=channels_2, out_channels=channels_3, name="tr2")
        self.st3 = Stage(
            num_channels=channels_3,
            num_modules=num_modules_3,
            num_filters=channels_3,
            has_se=self.has_se,
            name="st3")

        self.tr3 = TransitionLayer(
            in_channels=channels_3, out_channels=channels_4, name="tr3")
        self.st4 = Stage(
            num_channels=channels_4,
            num_modules=num_modules_4,
            num_filters=channels_4,
            has_se=self.has_se,
            name="st4")

        # classification
        num_filters_list = [32, 64, 128, 256]
        self.last_cls = LastClsOut(
            num_channel_list=channels_4,
            has_se=self.has_se,
            num_filters_list=num_filters_list,
            name="cls_head", )

        last_num_filters = [256, 512, 1024]
        self.cls_head_conv_list = []
        for idx in range(3):
            self.cls_head_conv_list.append(
                self.add_sublayer(
                    "cls_head_add{}".format(idx + 1),
                    ConvBNLayer(
                        num_channels=num_filters_list[idx] * 4,
                        num_filters=last_num_filters[idx],
                        filter_size=3,
                        stride=2,
                        name="cls_head_add" + str(idx + 1))))

        self.conv_last = ConvBNLayer(
            num_channels=1024,
            num_filters=2048,
            filter_size=1,
            stride=1,
            name="cls_head_last_conv")

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = Linear(
            2048,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, input):
        conv1 = self.conv_layer1_1(input)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.la1(conv2)

        tr1 = self.tr1([la1])
        st2 = self.st2(tr1)

        tr2 = self.tr2(st2)
        st3 = self.st3(tr2)

        tr3 = self.tr3(st3)
        st4 = self.st4(tr3)

        last_cls = self.last_cls(st4)

        y = last_cls[0]
        for idx in range(3):
            y = paddle.add(last_cls[idx + 1], self.cls_head_conv_list[idx](y))

        y = self.conv_last(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, y.shape[1]])
        y = self.out(y)
        return y


def HRNet_W18_C(**args):
    model = HRNet(width=18, **args)
    return model


def HRNet_W30_C(**args):
    model = HRNet(width=30, **args)
    return model


def HRNet_W32_C(**args):
    model = HRNet(width=32, **args)
    return model


def HRNet_W40_C(**args):
    model = HRNet(width=40, **args)
    return model


def HRNet_W44_C(**args):
    model = HRNet(width=44, **args)
    return model


def HRNet_W48_C(**args):
    model = HRNet(width=48, **args)
    return model


def HRNet_W60_C(**args):
    model = HRNet(width=60, **args)
    return model


def HRNet_W64_C(**args):
    model = HRNet(width=64, **args)
    return model


def SE_HRNet_W18_C(**args):
    model = HRNet(width=18, has_se=True, **args)
    return model


def SE_HRNet_W30_C(**args):
    model = HRNet(width=30, has_se=True, **args)
    return model


def SE_HRNet_W32_C(**args):
    model = HRNet(width=32, has_se=True, **args)
    return model


def SE_HRNet_W40_C(**args):
    model = HRNet(width=40, has_se=True, **args)
    return model


def SE_HRNet_W44_C(**args):
    model = HRNet(width=44, has_se=True, **args)
    return model


def SE_HRNet_W48_C(**args):
    model = HRNet(width=48, has_se=True, **args)
    return model


def SE_HRNet_W60_C(**args):
    model = HRNet(width=60, has_se=True, **args)
    return model


def SE_HRNet_W64_C(**args):
    model = HRNet(width=64, has_se=True, **args)
    return model
