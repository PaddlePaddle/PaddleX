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

from ...common.transformers.transformers import PretrainedConfig, PretrainedModel

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    # [k, in_c, out_c, s, use_se]
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [
        [3, 128, 256, 2, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
        [5, 256, 256, 1, False],
    ],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act == "relu6":
        return nn.ReLU6()
    else:
        raise RuntimeError("The activation function is not supported: {}".format(act))


class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if paddle.device.get_device().startswith("npu"):
            self.device = "npu"
        else:
            self.device = None

        if isinstance(self._output_size, int) and self._output_size == 1:
            self._gap = True
        elif (
            isinstance(self._output_size, tuple)
            and self._output_size[0] == 1
            and self._output_size[1] == 1
        ):
            self._gap = True
        else:
            self._gap = False

    def forward(self, x):
        if self.device == "npu" and self._gap:
            # Global Average Pooling
            N, C, _, _ = x.shape
            x_mean = paddle.mean(x, axis=[2, 3])
            x_mean = paddle.reshape(x_mean, [N, C, 1, 1])
            return x_mean
        else:
            return F.adaptive_avg_pool2d(
                x,
                output_size=self._output_size,
                data_format=self._data_format,
                name=self._name,
            )


class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels,
        filter_size,
        num_filters,
        stride,
        num_groups=1,
        lr_mult=1.0,
        act="hardswish",
    ):
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            weight_attr=ParamAttr(initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False,
        )

        self.bn = nn.BatchNorm2D(
            num_filters,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
        )
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparable(nn.Layer):
    def __init__(
        self,
        num_channels,
        num_filters,
        stride,
        dw_size=3,
        use_se=False,
        lr_mult=1.0,
        act="hardswish",
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_channels,
            filter_size=dw_size,
            stride=stride,
            num_groups=num_channels,
            lr_mult=lr_mult,
            act=act,
        )
        if use_se:
            self.se = SEModule(num_channels, lr_mult=lr_mult)
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            lr_mult=lr_mult,
            act=act,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
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


class PPLCNet(PretrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        config_dict = config.to_dict()
        self.scale = config["Global"]["scale"]
        self.class_num = config["Global"]["num_classes"]
        self.dropout_prob = config["Global"]["dropout_prob"]
        self.class_expand = config["Global"]["class_expand"]
        self.stride_list = config["Global"]["stride_list"]
        self.use_last_conv = config["Global"]["use_last_conv"]
        self.act = config["Global"]["act"]
        self.lr_mult_list = config["Global"]["lr_mult_list"]

        self.net_config = NET_CONFIG

        if isinstance(self.lr_mult_list, str):
            self.lr_mult_list = eval(self.lr_mult_list)

        assert isinstance(
            self.lr_mult_list, (list, tuple)
        ), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list)
        )
        assert (
            len(self.lr_mult_list) == 6
        ), "lr_mult_list length should be 6 but got {}".format(len(self.lr_mult_list))

        assert isinstance(
            self.stride_list, (list, tuple)
        ), "stride_list should be in (list, tuple) but got {}".format(
            type(self.stride_list)
        )
        assert (
            len(self.stride_list) == 5
        ), "stride_list length should be 5 but got {}".format(len(self.stride_list))

        for i, stride in enumerate(self.stride_list[1:]):
            self.net_config["blocks{}".format(i + 3)][0][3] = stride
        self.conv1 = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=make_divisible(16 * self.scale),
            stride=self.stride_list[0],
            lr_mult=self.lr_mult_list[0],
            act=self.act,
        )

        self.blocks2 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * self.scale),
                    num_filters=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    lr_mult=self.lr_mult_list[1],
                    act=self.act,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks2"])
            ]
        )

        self.blocks3 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * self.scale),
                    num_filters=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    lr_mult=self.lr_mult_list[2],
                    act=self.act,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks3"])
            ]
        )

        self.blocks4 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * self.scale),
                    num_filters=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    lr_mult=self.lr_mult_list[3],
                    act=self.act,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks4"])
            ]
        )

        self.blocks5 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * self.scale),
                    num_filters=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    lr_mult=self.lr_mult_list[4],
                    act=self.act,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks5"])
            ]
        )

        self.blocks6 = nn.Sequential(
            *[
                DepthwiseSeparable(
                    num_channels=make_divisible(in_c * self.scale),
                    num_filters=make_divisible(out_c * self.scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    lr_mult=self.lr_mult_list[5],
                    act=self.act,
                )
                for i, (k, in_c, out_c, s, se) in enumerate(self.net_config["blocks6"])
            ]
        )

        self.avg_pool = AdaptiveAvgPool2D(1)
        if self.use_last_conv:
            self.last_conv = nn.Conv2D(
                in_channels=make_divisible(
                    self.net_config["blocks6"][-1][2] * self.scale
                ),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            )
            self.act = _create_act(self.act)
            self.dropout = nn.Dropout(p=self.dropout_prob, mode="downscale_in_infer")
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = nn.Linear(
            (
                self.class_expand
                if self.use_last_conv
                else make_divisible(self.net_config["blocks6"][-1][2] * self.scale)
            ),
            self.class_num,
        )
        self.out_act = nn.Softmax(axis=-1)

    def forward(self, x):

        x = paddle.to_tensor(x[0])

        x = self.conv1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)

        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        x = self.out_act(x)

        return [x.cpu().numpy()]

    def get_transpose_weight_keys(self):
        t_layers = ["fc"]
        keys = []
        for key, _ in self.get_hf_state_dict().items():
            for t_layer in t_layers:
                if t_layer in key and key.endswith("weight"):
                    keys.append(key)
        return keys

    def get_hf_state_dict(self, *args, **kwargs):

        model_state_dict = self.state_dict(*args, **kwargs)

        hf_state_dict = {}
        for old_key, value in model_state_dict.items():
            if "_mean" in old_key:
                new_key = old_key.replace("_mean", "running_mean")
            elif "_variance" in old_key:
                new_key = old_key.replace("_variance", "running_var")
            else:
                new_key = old_key
            hf_state_dict[new_key] = value

        return hf_state_dict

    def set_hf_state_dict(self, state_dict, *args, **kwargs):

        key_mapping = {}
        for old_key in list(state_dict.keys()):
            if "running_mean" in old_key:
                key_mapping[old_key] = old_key.replace("running_mean", "_mean")
            elif "running_var" in old_key:
                key_mapping[old_key] = old_key.replace("running_var", "_variance")

        for old_key, new_key in key_mapping.items():
            state_dict[new_key] = state_dict.pop(old_key)

        return self.set_state_dict(state_dict, *args, **kwargs)
