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

from typing import Any, List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
from paddle.regularizer import L2Decay

from ....utils.benchmark import add_inference_operations, benchmark
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config import PPLCNetConfig


def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure the number of channels is a multiple of the specified divisor (common optimization for mobile networks)

    Args:
        v: Original number of channels
        divisor: Divisor, default 8
        min_value: Minimum number of channels, default None (takes divisor)

    Returns:
        Adjusted number of channels (integer)
    """

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act: str) -> nn.Layer:
    """
    Create activation function layer

    Args:
        act: Activation function name, supports "hardswish" / "relu" / "relu6"

    Returns:
        Activation function layer instance

    Raises:
        RuntimeError: Unsupported activation function type
    """
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act == "relu6":
        return nn.ReLU6()
    else:
        raise RuntimeError("The activation function is not supported: {}".format(act))


class AdaptiveAvgPool2D(nn.AdaptiveAvgPool2D):
    """
    AdaptiveAvgPool2D: : Adaptive average pooling layer optimized

    Args:
        *args: Positional arguments passed to parent class nn.AdaptiveAvgPool2D
        **kwargs: Keyword arguments passed to parent class nn.AdaptiveAvgPool2D

    Returns:
        paddle.Tensor: Pooled tensor with shape [N, C, 1, 1] (global pooling) or specified output size
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
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
    """
    ConvBNLayer: Combination layer of convolution, batch normalization and activation function

    Args:
        num_channels (int): Number of input channels
        filter_size (int): Kernel size of convolution layer
        num_filters (int): Number of output channels
        stride (int): Stride of convolution layer
        num_groups (int): Number of groups for grouped convolution, default 1
        lr_mult (float): Learning rate multiplier for layer parameters, default 1.0
        act (str): Activation function type, default "hardswish"

    Returns:
        paddle.Tensor: Output tensor after convolution + batch normalization + activation
    """

    def __init__(
        self,
        num_channels: int,
        filter_size: int,
        num_filters: int,
        stride: int,
        num_groups: int = 1,
        lr_mult: float = 1.0,
        act: str = "hardswish",
    ) -> None:
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

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DepthwiseSeparable(nn.Layer):
    """
    DepthwiseSeparable: Depthwise separable convolution layer with optional SE attention module

    Args:
        num_channels (int): Number of input channels
        num_filters (int): Number of output channels
        stride (int): Stride of depthwise convolution layer
        dw_size (int): Kernel size of depthwise convolution, default 3
        use_se (bool): Whether to use SE attention module, default False
        lr_mult (float): Learning rate multiplier for layer parameters, default 1.0
        act (str): Activation function type, default "hardswish"

    Returns:
        paddle.Tensor: Output tensor after depthwise separable convolution
    """

    def __init__(
        self,
        num_channels: int,
        num_filters: int,
        stride: int,
        reduction: int,
        dw_size: int,
        use_se: bool,
        lr_mult: float,
        act: str,
    ) -> None:
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
        self.se = (
            SEModule(num_channels, reduction, lr_mult) if use_se else nn.Identity()
        )
        self.pw_conv = ConvBNLayer(
            num_channels=num_channels,
            filter_size=1,
            num_filters=num_filters,
            stride=1,
            lr_mult=lr_mult,
            act=act,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.dw_conv(x)
        x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEModule(nn.Layer):
    """
    SEModule: Squeeze-and-Excitation attention module for channel-wise feature recalibration

    Args:
        channel (int): Number of input channels
        reduction (int): Channel reduction ratio for SE module, default 4
        lr_mult (float): Learning rate multiplier for module parameters, default 1.0

    Returns:
        paddle.Tensor: Attention-weighted tensor after SE module processing
    """

    def __init__(self, channel: int, reduction: int, lr_mult: float = 1.0) -> None:
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        conv_kwargs = {
            "kernel_size": 1,
            "stride": 1,
            "padding": 0,
            "weight_attr": ParamAttr(learning_rate=lr_mult),
            "bias_attr": ParamAttr(learning_rate=lr_mult),
        }
        self.conv1 = nn.Conv2D(channel, channel // reduction, **conv_kwargs)
        self.conv2 = nn.Conv2D(channel // reduction, channel, **conv_kwargs)
        self.relu = nn.ReLU()
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class PPLCNet(BatchNormHFStateDictMixin, PretrainedModel):
    """
    PPLCNet: Lightweight convolutional neural network for image classification tasks

    Args:
        config (PPLCNetConfig): Configuration instance containing model hyperparameters
            - scale (float): Channel scale factor for network width adjustment
            - class_num (int): Number of classification categories
            - dropout_prob (float): Dropout probability for last convolution layer
            - class_expand (int): Expansion channel number for last convolution layer
            - stride_list (List[int]): Stride list for different blocks, length must be 5
            - use_last_conv (bool): Whether to use last convolution layer before fc
            - act (str): Activation function type used in network
            - lr_mult_list (List[float]): Learning rate multipliers for different layers, length must be 6
            - net_config (Dict[str, Any]): Network configuration dict containing block parameters

    Returns:
        List[numpy.ndarray]: List containing classification probability numpy array
    """

    config_class = PPLCNetConfig

    def __init__(self, config: PPLCNetConfig) -> None:
        super().__init__(config)

        self.scale = config.scale
        self.class_num = config.class_num
        self.dropout_prob = config.dropout_prob
        self.class_expand = config.class_expand
        self.stride_list = config.stride_list
        self.reduction = config.reduction
        self.use_last_conv = config.use_last_conv
        self.act = config.act
        self.lr_mult_list = (
            eval(config.lr_mult_list)
            if isinstance(config.lr_mult_list, str)
            else config.lr_mult_list
        )
        self.net_config = config.net_config

        assert isinstance(
            self.lr_mult_list, (list, tuple)
        ), f"lr_mult_list should be in (list, tuple) but got {type(self.lr_mult_list)}"
        assert (
            len(self.lr_mult_list) == 6
        ), f"lr_mult_list length should be 6 but got {len(self.lr_mult_list)}"
        assert isinstance(
            self.stride_list, (list, tuple)
        ), f"stride_list should be in (list, tuple) but got {type(self.stride_list)}"
        assert (
            len(self.stride_list) == 5
        ), f"stride_list length should be 5 but got {len(self.stride_list)}"

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

        def _build_block(block_name, lr_idx):
            return nn.Sequential(
                *[
                    DepthwiseSeparable(
                        num_channels=make_divisible(in_c * self.scale),
                        num_filters=make_divisible(out_c * self.scale),
                        dw_size=k,
                        stride=s,
                        reduction=self.reduction,
                        use_se=se,
                        lr_mult=self.lr_mult_list[lr_idx],
                        act=self.act,
                    )
                    for i, (k, in_c, out_c, s, se) in enumerate(
                        self.net_config[block_name]
                    )
                ]
            )

        self.blocks2 = _build_block("blocks2", 1)
        self.blocks3 = _build_block("blocks3", 2)
        self.blocks4 = _build_block("blocks4", 3)
        self.blocks5 = _build_block("blocks5", 4)
        self.blocks6 = _build_block("blocks6", 5)

        self.avg_pool = AdaptiveAvgPool2D(1)
        self.last_conv = None
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

        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        if self.use_last_conv:
            fc_in_channels = self.class_expand
        else:
            fc_in_channels = make_divisible(
                self.net_config["blocks6"][-1][2] * self.scale
            )
        self.fc = nn.Linear(fc_in_channels, self.class_num)
        self.out_act = nn.Softmax(axis=-1)

    add_inference_operations("pplcnet_forward")

    @benchmark.timeit_with_options(name="pplcnet_forward")
    def forward(self, x: List) -> List:

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
