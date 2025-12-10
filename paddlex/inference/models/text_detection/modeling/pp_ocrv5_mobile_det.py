# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config import PPOCRV5MobileDetConfig
from .pp_ocrv5_modules import DBHead, LearnableAffineBlock

NET_CONFIG = {
    "blocks2":
    # k, in_c, out_c, s, use_se
    [[3, 16, 24, 1, False]],
    "blocks3": [
        [3, 24, 48, 2, False],
        [3, 48, 48, 1, False],
    ],
    "blocks4": [
        [3, 48, 96, 2, False],
        [3, 96, 96, 1, False],
    ],
    "blocks5": [
        [3, 96, 192, 2, False],
        [5, 192, 192, 1, False],
        [5, 192, 192, 1, False],
        [5, 192, 192, 1, False],
        [5, 192, 192, 1, False],
    ],
    "blocks6": [
        [5, 192, 384, 2, True],
        [5, 384, 384, 1, True],
        [5, 384, 384, 1, False],
        [5, 384, 384, 1, False],
    ],
    "layer_list_out_channels": [12, 18, 42, 360],
}


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Act(nn.Layer):
    def __init__(self, act="hswish", lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        else:
            assert act == "relu"
            self.act = nn.ReLU()
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        return self.lab(self.act(x))


class ConvBNLayer(nn.Layer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups=1, lr_mult=1.0
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False,
        )

        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LearnableRepLayer(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        num_conv_branches=1,
        lr_mult=1.0,
        lab_lr=0.1,
    ):
        super().__init__()
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = (
            nn.BatchNorm2D(
                num_features=in_channels,
                weight_attr=ParamAttr(learning_rate=lr_mult),
                bias_attr=ParamAttr(learning_rate=lr_mult),
            )
            if out_channels == in_channels and stride == 1
            else None
        )

        self.conv_kxk = nn.LayerList(
            [
                ConvBNLayer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    lr_mult=lr_mult,
                )
                for _ in range(self.num_conv_branches)
            ]
        )

        self.conv_1x1 = (
            ConvBNLayer(
                in_channels, out_channels, 1, stride, groups=groups, lr_mult=lr_mult
            )
            if kernel_size > 1
            else None
        )

        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = Act(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):

        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        if "npu" in paddle.device.get_device():
            self.avg_pool = nn.MeanPool2D(1, 1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class LCNetV3Block(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        conv_kxk_num=4,
        lr_mult=1.0,
        lab_lr=0.1,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )
        if use_se:
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPLCNetV3(nn.Layer):
    def __init__(
        self,
        scale=1.0,
        conv_kxk_num=4,
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        lab_lr=0.1,
        net_config=NET_CONFIG,
        out_channels=512,
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.lr_mult_list = lr_mult_list

        self.net_config = net_config
        self.out_channels = make_divisible(out_channels * scale)

        assert isinstance(
            self.lr_mult_list, (list, tuple)
        ), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list)
        )
        assert (
            len(self.lr_mult_list) == 6
        ), "lr_mult_list length should be 6 but got {}".format(len(self.lr_mult_list))

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=make_divisible(16 * scale),
            kernel_size=3,
            stride=2,
            lr_mult=self.lr_mult_list[0],
        )

        self.blocks2 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[1],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[2],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[3],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[4],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                LCNetV3Block(
                    in_channels=make_divisible(in_c * scale),
                    out_channels=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                    lr_mult=self.lr_mult_list[5],
                    lab_lr=lab_lr,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
            ]
        )

        mv_c = self.net_config["layer_list_out_channels"]  # [12, 18, 42, 360]

        self.out_channels = [
            make_divisible(self.net_config["blocks3"][-1][2] * scale),
            make_divisible(self.net_config["blocks4"][-1][2] * scale),
            make_divisible(self.net_config["blocks5"][-1][2] * scale),
            make_divisible(self.net_config["blocks6"][-1][2] * scale),
        ]

        self.layer_list = nn.LayerList(
            [
                nn.Conv2D(self.out_channels[0], int(mv_c[0] * scale), 1, 1, 0),
                nn.Conv2D(self.out_channels[1], int(mv_c[1] * scale), 1, 1, 0),
                nn.Conv2D(self.out_channels[2], int(mv_c[2] * scale), 1, 1, 0),
                nn.Conv2D(self.out_channels[3], int(mv_c[3] * scale), 1, 1, 0),
            ]
        )
        self.out_channels = [
            int(mv_c[0] * scale),
            int(mv_c[1] * scale),
            int(mv_c[2] * scale),
            int(mv_c[3] * scale),
        ]

    def forward(self, x):
        out_list = []
        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)
        x = self.blocks4(x)
        out_list.append(x)
        x = self.blocks5(x)
        out_list.append(x)
        x = self.blocks6(x)
        out_list.append(x)

        out_list[0] = self.layer_list[0](out_list[0])
        out_list[1] = self.layer_list[1](out_list[1])
        out_list[2] = self.layer_list[2](out_list[2])
        out_list[3] = self.layer_list[3](out_list[3])
        return out_list


class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        if "npu" in paddle.device.get_device():
            self.avg_pool = nn.MeanPool2D(1, 1)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


class RSELayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, shortcut=True):
        super(RSELayer, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False,
        )
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins):
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, shortcut=True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut)
            )
            self.inp_conv.append(
                RSELayer(
                    out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut
                )
            )

    def forward(self, x):
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/16
        out3 = in3 + F.upsample(
            out4, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/8
        out2 = in2 + F.upsample(
            out3, scale_factor=2, mode="nearest", align_mode=1
        )  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


class PPOCRV5MobileDet(BatchNormHFStateDictMixin, PretrainedModel):
    config_class = PPOCRV5MobileDetConfig

    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__(config)

        self.backbone_scale = config.backbone_scale
        self.backbone_det = config.backbone_det
        self.backbone_conv_kxk_num = config.backbone_conv_kxk_num
        self.backbone_lr_mult_list = config.backbone_lr_mult_list
        self.backbone_lab_lr = config.backbone_lab_lr
        self.backbone_net_config = config.backbone_net_config
        self.backbone_out_channels = config.backbone_out_channels

        self.neck_out_channels = config.neck_out_channels
        self.neck_shortcut = config.neck_shortcut

        self.head_k = config.head_k
        self.head_fix_nan = config.head_fix_nan

        self.backbone = PPLCNetV3(
            scale=self.backbone_scale,
            conv_kxk_num=self.backbone_conv_kxk_num,
            lr_mult_list=self.backbone_lr_mult_list,
            lab_lr=self.backbone_lab_lr,
            net_config=self.backbone_net_config,
            out_channels=self.backbone_out_channels,
        )

        neck_in_channels = self.backbone.out_channels
        self.neck = RSEFPN(
            in_channels=neck_in_channels,
            out_channels=self.neck_out_channels,
            shortcut=self.neck_shortcut,
        )

        head_in_channels = self.neck_out_channels
        self.head = DBHead(
            in_channels=head_in_channels, k=self.head_k, fix_nan=self.head_fix_nan
        )

    def forward(self, x):

        x = paddle.to_tensor(x[0])

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return [x.cpu().numpy()]
