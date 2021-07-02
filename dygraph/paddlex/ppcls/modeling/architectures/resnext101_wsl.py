import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

__all__ = [
    "ResNeXt101_32x8d_wsl", "ResNeXt101_32x16d_wsl", "ResNeXt101_32x32d_wsl",
    "ResNeXt101_32x48d_wsl"
]


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        if "downsample" in name:
            conv_name = name + ".0"
        else:
            conv_name = name
        self._conv = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(name=conv_name + ".weight"),
            bias_attr=False)
        if "downsample" in name:
            bn_name = name[:9] + "downsample.1"
        else:
            if "conv1" == name:
                bn_name = "bn" + name[-1]
            else:
                bn_name = (name[:10] if name[7:9].isdigit() else name[:9]
                           ) + "bn" + name[-1]
        self._bn = BatchNorm(
            num_channels=output_channels,
            act=act,
            param_attr=ParamAttr(name=bn_name + ".weight"),
            bias_attr=ParamAttr(name=bn_name + ".bias"),
            moving_mean_name=bn_name + ".running_mean",
            moving_variance_name=bn_name + ".running_var")

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x


class ShortCut(nn.Layer):
    def __init__(self, input_channels, output_channels, stride, name=None):
        super(ShortCut, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        if input_channels != output_channels or stride != 1:
            self._conv = ConvBNLayer(
                input_channels,
                output_channels,
                filter_size=1,
                stride=stride,
                name=name)

    def forward(self, inputs):
        if self.input_channels != self.output_channels or self.stride != 1:
            return self._conv(inputs)
        return inputs


class BottleneckBlock(nn.Layer):
    def __init__(self, input_channels, output_channels, stride, cardinality,
                 width, name):
        super(BottleneckBlock, self).__init__()

        self._conv0 = ConvBNLayer(
            input_channels,
            output_channels,
            filter_size=1,
            act="relu",
            name=name + ".conv1")
        self._conv1 = ConvBNLayer(
            output_channels,
            output_channels,
            filter_size=3,
            act="relu",
            stride=stride,
            groups=cardinality,
            name=name + ".conv2")
        self._conv2 = ConvBNLayer(
            output_channels,
            output_channels // (width // 8),
            filter_size=1,
            act=None,
            name=name + ".conv3")
        self._short = ShortCut(
            input_channels,
            output_channels // (width // 8),
            stride=stride,
            name=name + ".downsample")

    def forward(self, inputs):
        x = self._conv0(inputs)
        x = self._conv1(x)
        x = self._conv2(x)
        y = self._short(inputs)
        y = paddle.add(x, y)
        y = F.relu(y)
        return y


class ResNeXt101WSL(nn.Layer):
    def __init__(self, layers=101, cardinality=32, width=48, class_dim=1000):
        super(ResNeXt101WSL, self).__init__()

        self.class_dim = class_dim

        self.layers = layers
        self.cardinality = cardinality
        self.width = width
        self.scale = width // 8

        self.depth = [3, 4, 23, 3]
        self.base_width = cardinality * width
        num_filters = [self.base_width * i
                       for i in [1, 2, 4, 8]]  # [256, 512, 1024, 2048]
        self._conv_stem = ConvBNLayer(
            3, 64, 7, stride=2, act="relu", name="conv1")
        self._pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self._conv1_0 = BottleneckBlock(
            64,
            num_filters[0],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer1.0")
        self._conv1_1 = BottleneckBlock(
            num_filters[0] // (width // 8),
            num_filters[0],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer1.1")
        self._conv1_2 = BottleneckBlock(
            num_filters[0] // (width // 8),
            num_filters[0],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer1.2")

        self._conv2_0 = BottleneckBlock(
            num_filters[0] // (width // 8),
            num_filters[1],
            stride=2,
            cardinality=self.cardinality,
            width=self.width,
            name="layer2.0")
        self._conv2_1 = BottleneckBlock(
            num_filters[1] // (width // 8),
            num_filters[1],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer2.1")
        self._conv2_2 = BottleneckBlock(
            num_filters[1] // (width // 8),
            num_filters[1],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer2.2")
        self._conv2_3 = BottleneckBlock(
            num_filters[1] // (width // 8),
            num_filters[1],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer2.3")

        self._conv3_0 = BottleneckBlock(
            num_filters[1] // (width // 8),
            num_filters[2],
            stride=2,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.0")
        self._conv3_1 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.1")
        self._conv3_2 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.2")
        self._conv3_3 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.3")
        self._conv3_4 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.4")
        self._conv3_5 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.5")
        self._conv3_6 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.6")
        self._conv3_7 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.7")
        self._conv3_8 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.8")
        self._conv3_9 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.9")
        self._conv3_10 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.10")
        self._conv3_11 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.11")
        self._conv3_12 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.12")
        self._conv3_13 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.13")
        self._conv3_14 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.14")
        self._conv3_15 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.15")
        self._conv3_16 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.16")
        self._conv3_17 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.17")
        self._conv3_18 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.18")
        self._conv3_19 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.19")
        self._conv3_20 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.20")
        self._conv3_21 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.21")
        self._conv3_22 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[2],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer3.22")

        self._conv4_0 = BottleneckBlock(
            num_filters[2] // (width // 8),
            num_filters[3],
            stride=2,
            cardinality=self.cardinality,
            width=self.width,
            name="layer4.0")
        self._conv4_1 = BottleneckBlock(
            num_filters[3] // (width // 8),
            num_filters[3],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer4.1")
        self._conv4_2 = BottleneckBlock(
            num_filters[3] // (width // 8),
            num_filters[3],
            stride=1,
            cardinality=self.cardinality,
            width=self.width,
            name="layer4.2")

        self._avg_pool = AdaptiveAvgPool2D(1)
        self._out = Linear(
            num_filters[3] // (width // 8),
            class_dim,
            weight_attr=ParamAttr(name="fc.weight"),
            bias_attr=ParamAttr(name="fc.bias"))

    def forward(self, inputs):
        x = self._conv_stem(inputs)
        x = self._pool(x)

        x = self._conv1_0(x)
        x = self._conv1_1(x)
        x = self._conv1_2(x)

        x = self._conv2_0(x)
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        x = self._conv2_3(x)

        x = self._conv3_0(x)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        x = self._conv3_3(x)
        x = self._conv3_4(x)
        x = self._conv3_5(x)
        x = self._conv3_6(x)
        x = self._conv3_7(x)
        x = self._conv3_8(x)
        x = self._conv3_9(x)
        x = self._conv3_10(x)
        x = self._conv3_11(x)
        x = self._conv3_12(x)
        x = self._conv3_13(x)
        x = self._conv3_14(x)
        x = self._conv3_15(x)
        x = self._conv3_16(x)
        x = self._conv3_17(x)
        x = self._conv3_18(x)
        x = self._conv3_19(x)
        x = self._conv3_20(x)
        x = self._conv3_21(x)
        x = self._conv3_22(x)

        x = self._conv4_0(x)
        x = self._conv4_1(x)
        x = self._conv4_2(x)

        x = self._avg_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self._out(x)
        return x


def ResNeXt101_32x8d_wsl(**args):
    model = ResNeXt101WSL(cardinality=32, width=8, **args)
    return model


def ResNeXt101_32x16d_wsl(**args):
    model = ResNeXt101WSL(cardinality=32, width=16, **args)
    return model


def ResNeXt101_32x32d_wsl(**args):
    model = ResNeXt101WSL(cardinality=32, width=32, **args)
    return model


def ResNeXt101_32x48d_wsl(**args):
    model = ResNeXt101WSL(cardinality=32, width=48, **args)
    return model
