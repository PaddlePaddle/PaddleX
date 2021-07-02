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
"""
    MixNet for ImageNet-1K, implemented in Paddle.
    Original paper: 'MixConv: Mixed Depthwise Convolutional Kernels,'
    https://arxiv.org/abs/1907.09595.
"""

__all__ = ['MixNet_S', 'MixNet_M', 'MixNet_L']

import os
from inspect import isfunction
from functools import reduce
import paddle
import paddle.nn as nn


class Identity(nn.Layer):
    """
    Identity block.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def round_channels(channels, divisor=8):
    """
    Round weighted channel number (make divisible operation).

    Parameters:
    ----------
    channels : int or float
        Original number of channels.
    divisor : int, default 8
        Alignment value.

    Returns:
    -------
    int
        Weighted number of channels.
    """
    rounded_channels = max(
        int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def get_activation_layer(activation):
    """
    Create activation layer from string/function.

    Parameters:
    ----------
    activation : function, or str, or nn.Module
        Activation function or name of activation function.

    Returns:
    -------
    nn.Module
        Activation layer.
    """
    assert activation is not None
    if isfunction(activation):
        return activation()
    elif isinstance(activation, str):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "relu6":
            return nn.ReLU6()
        elif activation == "swish":
            return nn.Swish()
        elif activation == "hswish":
            return nn.Hardswish()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "hsigmoid":
            return nn.Hardsigmoid()
        elif activation == "identity":
            return Identity()
        else:
            raise NotImplementedError()
    else:
        assert isinstance(activation, nn.Layer)
        return activation


class ConvBlock(nn.Layer):
    """
    Standard convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=nn.ReLU()):
        super(ConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_pad = (isinstance(padding, (list, tuple)) and
                        (len(padding) == 4))

        if self.use_pad:
            self.pad = padding
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
            weight_attr=None)
        if self.use_bn:
            self.bn = nn.BatchNorm2D(num_features=out_channels, epsilon=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


class SEBlock(nn.Layer):
    def __init__(self,
                 channels,
                 reduction=16,
                 mid_channels=None,
                 round_mid=False,
                 use_conv=True,
                 mid_activation=nn.ReLU(),
                 out_activation=nn.Sigmoid()):
        super(SEBlock, self).__init__()
        self.use_conv = use_conv
        if mid_channels is None:
            mid_channels = channels // reduction if not round_mid else round_channels(
                float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2D(output_size=1)
        if use_conv:
            self.conv1 = nn.Conv2D(
                in_channels=channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias_attr=True,
                weight_attr=None)

        else:
            self.fc1 = nn.Linear(
                in_features=channels, out_features=mid_channels)
        self.activ = get_activation_layer(mid_activation)
        if use_conv:
            self.conv2 = nn.Conv2D(
                in_channels=mid_channels,
                out_channels=channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias_attr=True,
                weight_attr=None)
        else:
            self.fc2 = nn.Linear(
                in_features=mid_channels, out_features=channels)
        self.sigmoid = get_activation_layer(out_activation)

    def forward(self, x):
        w = self.pool(x)
        if not self.use_conv:
            w = w.reshape(shape=[w.shape[0], -1])
        w = self.conv1(w) if self.use_conv else self.fc1(w)
        w = self.activ(w)
        w = self.conv2(w) if self.use_conv else self.fc2(w)
        w = self.sigmoid(w)
        if not self.use_conv:
            w = w.unsqueeze(2).unsqueeze(3)
        x = x * w
        return x


class MixConv(nn.Layer):
    """
    Mixed convolution layer from 'MixConv: Mixed Depthwise Convolutional Kernels,'
    https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 axis=1):
        super(MixConv, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size,
                                                list) else [kernel_size]
        padding = padding if isinstance(padding, list) else [padding]
        kernel_count = len(kernel_size)
        self.splitted_in_channels = self.split_channels(in_channels,
                                                        kernel_count)
        splitted_out_channels = self.split_channels(out_channels, kernel_count)
        for i, kernel_size_i in enumerate(kernel_size):
            in_channels_i = self.splitted_in_channels[i]
            out_channels_i = splitted_out_channels[i]
            padding_i = padding[i]
            _ = self.add_sublayer(
                name=str(i),
                sublayer=nn.Conv2D(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    kernel_size=kernel_size_i,
                    stride=stride,
                    padding=padding_i,
                    dilation=dilation,
                    groups=(out_channels_i
                            if out_channels == groups else groups),
                    bias_attr=bias,
                    weight_attr=None))
        self.axis = axis

    def forward(self, x):
        xx = paddle.split(x, self.splitted_in_channels, axis=self.axis)
        xx = paddle.split(x, self.splitted_in_channels, axis=self.axis)
        out = [
            conv_i(x_i) for x_i, conv_i in zip(xx, self._sub_layers.values())
        ]
        x = paddle.concat(tuple(out), axis=self.axis)
        return x

    @staticmethod
    def split_channels(channels, kernel_count):
        splitted_channels = [channels // kernel_count] * kernel_count
        splitted_channels[0] += channels - sum(splitted_channels)
        return splitted_channels


class MixConvBlock(nn.Layer):
    """
    Mixed convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU()
        Activation function or name of activation function.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=nn.ReLU()):
        super(MixConvBlock, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn

        self.conv = MixConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2D(num_features=out_channels, epsilon=bn_eps)
        if self.activate:
            self.activ = get_activation_layer(activation)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def mixconv1x1_block(in_channels,
                     out_channels,
                     kernel_count,
                     stride=1,
                     groups=1,
                     bias=False,
                     use_bn=True,
                     bn_eps=1e-5,
                     activation=nn.ReLU()):
    """
    1x1 version of the mixed convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_count : int
        Kernel count.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str, or None, default nn.ReLU()
        Activation function or name of activation function.
    """
    return MixConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=([1] * kernel_count),
        stride=stride,
        padding=([0] * kernel_count),
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class MixUnit(nn.Layer):
    """
    MixNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.  exp_channels : int
        Number of middle (expanded) channels.
    stride : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_kernel_count : int
        Expansion convolution kernel count for each unit.
    conv1_kernel_count : int
        Conv1 kernel count for each unit.
    conv2_kernel_count : int
        Conv2 kernel count for each unit.
    exp_factor : int
        Expansion factor for each unit.
    se_factor : int
        SE reduction factor for each unit.
    activation : str
        Activation function or name of activation function.
    """

    def __init__(self, in_channels, out_channels, stride, exp_kernel_count,
                 conv1_kernel_count, conv2_kernel_count, exp_factor, se_factor,
                 activation):
        super(MixUnit, self).__init__()
        assert exp_factor >= 1
        assert se_factor >= 0
        self.residual = (in_channels == out_channels) and (stride == 1)
        self.use_se = se_factor > 0
        mid_channels = exp_factor * in_channels
        self.use_exp_conv = exp_factor > 1

        if self.use_exp_conv:
            if exp_kernel_count == 1:
                self.exp_conv = ConvBlock(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    bias=False,
                    use_bn=True,
                    bn_eps=1e-5,
                    activation=activation)
            else:
                self.exp_conv = mixconv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_count=exp_kernel_count,
                    activation=activation)
        if conv1_kernel_count == 1:
            self.conv1 = ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                groups=mid_channels,
                bias=False,
                use_bn=True,
                bn_eps=1e-5,
                activation=activation)
        else:
            self.conv1 = MixConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=[3 + 2 * i for i in range(conv1_kernel_count)],
                stride=stride,
                padding=[1 + i for i in range(conv1_kernel_count)],
                groups=mid_channels,
                activation=activation)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                reduction=(exp_factor * se_factor),
                round_mid=False,
                mid_activation=activation)
        if conv2_kernel_count == 1:
            self.conv2 = ConvBlock(
                in_channels=mid_channels,
                out_channels=out_channels,
                activation=None,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                use_bn=True,
                bn_eps=1e-5)
        else:
            self.conv2 = mixconv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_count=conv2_kernel_count,
                activation=None)

    def forward(self, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MixInitBlock(nn.Layer):
    """
    MixNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(MixInitBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            kernel_size=3,
            padding=1)
        self.conv2 = MixUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            stride=1,
            exp_kernel_count=1,
            conv1_kernel_count=1,
            conv2_kernel_count=1,
            exp_factor=1,
            se_factor=0,
            activation="relu")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MixNet(nn.Layer):
    """
    MixNet model from 'MixConv: Mixed Depthwise Convolutional Kernels,'
    https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    exp_kernel_counts : list of list of int
        Expansion convolution kernel count for each unit.
    conv1_kernel_counts : list of list of int
        Conv1 kernel count for each unit.
    conv2_kernel_counts : list of list of int
        Conv2 kernel count for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    class_dim : int, default 1000
        Number of classification classes.
    """

    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 exp_kernel_counts,
                 conv1_kernel_counts,
                 conv2_kernel_counts,
                 exp_factors,
                 se_factors,
                 in_channels=3,
                 in_size=(224, 224),
                 class_dim=1000):
        super(MixNet, self).__init__()
        self.in_size = in_size
        self.class_dim = class_dim

        self.features = nn.Sequential()
        self.features.add_sublayer(
            "init_block",
            MixInitBlock(
                in_channels=in_channels, out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if ((j == 0) and (i != 3)) or (
                    (j == len(channels_per_stage) // 2) and (i == 3)) else 1
                exp_kernel_count = exp_kernel_counts[i][j]
                conv1_kernel_count = conv1_kernel_counts[i][j]
                conv2_kernel_count = conv2_kernel_counts[i][j]
                exp_factor = exp_factors[i][j]
                se_factor = se_factors[i][j]
                activation = "relu" if i == 0 else "swish"
                stage.add_sublayer(
                    "unit{}".format(j + 1),
                    MixUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        exp_kernel_count=exp_kernel_count,
                        conv1_kernel_count=conv1_kernel_count,
                        conv2_kernel_count=conv2_kernel_count,
                        exp_factor=exp_factor,
                        se_factor=se_factor,
                        activation=activation))
                in_channels = out_channels
            self.features.add_sublayer("stage{}".format(i + 1), stage)
        self.features.add_sublayer(
            "final_block",
            ConvBlock(
                in_channels=in_channels,
                out_channels=final_block_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=False,
                use_bn=True,
                bn_eps=1e-5,
                activation=nn.ReLU()))
        in_channels = final_block_channels
        self.features.add_sublayer(
            "final_pool", nn.AvgPool2D(
                kernel_size=7, stride=1))

        self.output = nn.Linear(
            in_features=in_channels, out_features=class_dim)

    def forward(self, x):
        x = self.features(x)
        reshape_dim = reduce(lambda x, y: x * y, x.shape[1:])
        x = x.reshape(shape=[x.shape[0], reshape_dim])
        x = self.output(x)
        return x


def get_mixnet(version, width_scale, model_name=None, **kwargs):
    """
    Create MixNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('s' or 'm').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name.
    """

    if version == "s":
        init_block_channels = 16
        channels = [[24, 24], [40, 40, 40, 40], [80, 80, 80],
                    [120, 120, 120, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 1, 1],
                             [2, 2, 2, 1, 1, 1]]
        conv1_kernel_counts = [[1, 1], [3, 2, 2, 2], [3, 2, 2],
                               [3, 4, 4, 5, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [2, 2, 2],
                               [2, 2, 2, 1, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6], [6, 3, 3, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4], [2, 2, 2, 2, 2, 2]]
    elif version == "m":
        init_block_channels = 24
        channels = [[32, 32], [40, 40, 40, 40], [80, 80, 80, 80],
                    [120, 120, 120, 120, 200, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2],
                             [1, 2, 2, 2, 1, 1, 1, 1]]
        conv1_kernel_counts = [[3, 1], [4, 2, 2, 2], [3, 4, 4, 4],
                               [1, 4, 4, 4, 4, 4, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2],
                               [1, 2, 2, 2, 1, 2, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6, 6],
                       [6, 3, 3, 3, 6, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4, 4],
                      [2, 2, 2, 2, 2, 2, 2, 2]]
    else:
        raise ValueError("Unsupported MixNet version {}".format(version))

    final_block_channels = 1536

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci]
                    for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale)

    net = MixNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        exp_kernel_counts=exp_kernel_counts,
        conv1_kernel_counts=conv1_kernel_counts,
        conv2_kernel_counts=conv2_kernel_counts,
        exp_factors=exp_factors,
        se_factors=se_factors,
        **kwargs)

    return net


def MixNet_S(**kwargs):
    """
    MixNet-S model from 'MixConv: Mixed Depthwise Convolutional Kernels,'
    https://arxiv.org/abs/1907.09595.
    """
    return get_mixnet(
        version="s", width_scale=1.0, model_name="MixNet_S", **kwargs)


def MixNet_M(**kwargs):
    """
    MixNet-M model from 'MixConv: Mixed Depthwise Convolutional Kernels,'
    https://arxiv.org/abs/1907.09595.
    """
    return get_mixnet(
        version="m", width_scale=1.0, model_name="MixNet_M", **kwargs)


def MixNet_L(**kwargs):
    """
    MixNet-L model from 'MixConv: Mixed Depthwise Convolutional Kernels,'
    https://arxiv.org/abs/1907.09595.
    """
    return get_mixnet(
        version="m", width_scale=1.3, model_name="MixNet_L", **kwargs)
