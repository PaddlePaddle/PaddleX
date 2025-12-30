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
import math
from typing import Any, List, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant


def get_bias_attr(k: float) -> ParamAttr:
    """
    get_bias_attr

    Args:
        k (float): Scaling factor for standard deviation calculation

    Returns:
        ParamAttr: Parameter attribute with uniform initializer
    """
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class LearnableAffineBlock(nn.Layer):
    """
    LearnableAffineBlock

    Args:
        scale_value (float, optional): Initial value for scale parameter, default is 1.0
        bias_value (float, optional): Initial value for bias parameter, default is 0.0
        lr_mult (float, optional): Learning rate multiplier for base learning rate, default is 1.0
        lab_lr (float, optional): Additional learning rate multiplier for affine parameters, default is 0.01

    Returns:
        paddle.Tensor: Output tensor after affine transformation (scale * x + bias)
    """

    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        lr_mult: float = 1.0,
        lab_lr: float = 0.01,
    ) -> None:
        super().__init__()
        self.scale = self.create_parameter(
            shape=[
                1,
            ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        )
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[
                1,
            ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr),
        )
        self.add_parameter("bias", self.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.scale * x + self.bias


class Head(nn.Layer):
    """
    Head

    Args:
        in_channels (int): Number of input channels
        kernel_list (List[int]): List of kernel sizes for conv/transposed conv layers
        fix_nan (bool): Whether to fix NaN issues (unused in current implementation)
        **kwargs: Additional keyword arguments

    Returns:
        paddle.Tensor: 1-channel sigmoid output tensor (or tuple with feature tensor if return_f=True)
    """

    def __init__(
        self, in_channels: int, kernel_list: List[int], fix_nan: bool, **kwargs: Any
    ) -> None:
        super(Head, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False,
        )
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu",
        )

        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu",
        )
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4),
        )

        self.fix_nan = fix_nan

    def forward(
        self, x: paddle.Tensor, return_f: bool = False
    ) -> Union[paddle.Tensor, Tuple]:
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x


class DBHead(nn.Layer):
    """
    DBHead

    Paper: https://arxiv.org/abs/1911.08947

    Args:
        in_channels (int): Number of input channels
        k (int): DB head hyperparameter (kernel factor)
        **kwargs: Additional keyword arguments for Head class (kernel_list, fix_nan)

    Returns:
        paddle.Tensor: Shrinkage map tensor after DB binarization
    """

    def __init__(self, in_channels: int, k: int, **kwargs) -> None:
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = Head(in_channels, **kwargs)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shrink_maps = self.binarize(x)
        return shrink_maps
