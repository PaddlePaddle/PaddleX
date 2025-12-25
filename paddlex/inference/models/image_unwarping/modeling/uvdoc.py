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

from ....utils.benchmark import add_inference_operations, benchmark
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config import UVDocNetConfig


def conv3x3(
    in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
) -> nn.Conv2D:
    """
    conv3x3: 3x3 convolution layer with same padding

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of convolution layer
        stride (int, optional): Convolution stride, default is 1

    Returns:
        nn.Conv2D: Convolutional layer with same padding (padding = kernel_size // 2)
    """

    return nn.Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def dilated_conv_bn_act(
    in_channels: int, out_channels: int, dilation: int
) -> nn.Sequential:
    """
    dilated_conv_bn_act: Dilated convolution block with BN and ReLU activation

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        dilation (int): Dilation rate for dilated convolution

    Returns:
        nn.Sequential: Sequential block containing dilated conv, BN and ReLU layers
    """

    model = nn.Sequential(
        nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            bias_attr=False,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
        ),
        nn.BatchNorm2D(out_channels),
        nn.ReLU(),
    )
    return model


def dilated_conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dilation: int,
    stride: int = 1,
) -> nn.Sequential:
    """
    dilated_conv: Pure dilated convolution layer for multi-scale feature learning

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of dilated convolution
        dilation (int): Dilation rate for dilated convolution
        stride (int, optional): Convolution stride, default is 1

    Returns:
        nn.Sequential: Sequential block containing only the dilated convolution layer
    """
    model = nn.Sequential(
        nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size // 2),
            dilation=dilation,
        )
    )
    return model


class ResidualBlockWithDilation(nn.Layer):
    """
    ResidualBlockWithDilation: Residual block with optional dilated convolution and downsampling

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of convolution layers
        stride (int, optional): Convolution stride for first conv layer, default is 1
        downsample (Optional[nn.Layer]): Downsampling layer for residual connection, default is None
        is_activation (bool, optional): Whether to apply activation (unused in current implementation), default is True
        is_top (bool, optional): Whether it is the top block (uses standard conv instead of dilated conv), default is False

    Returns:
        paddle.Tensor: Output tensor after residual block with conv/dilated conv and ReLU activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        downsample: Optional[nn.Layer] = None,
        is_activation: bool = True,
        is_top: bool = False,
    ):
        super(ResidualBlockWithDilation, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.is_activation = is_activation
        self.is_top = is_top
        if self.stride != 1 or self.is_top:
            self.conv1 = conv3x3(in_channels, out_channels, kernel_size, self.stride)
            self.conv2 = conv3x3(out_channels, out_channels, kernel_size)
        else:
            self.conv1 = dilated_conv(
                in_channels, out_channels, kernel_size, dilation=3
            )
            self.conv2 = dilated_conv(
                out_channels, out_channels, kernel_size, dilation=3
            )
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.bn2(self.conv2(out1))
        out2 += residual
        out = self.relu(out2)
        return out


class ResnetStraight(nn.Layer):
    """
    ResnetStraight: Straightforward ResNet architecture with residual blocks and optional dilated convolution

    Args:
        num_filter (int): Base number of filters/channels for the network
        map_num (List[int]): List of channel scaling factors for each layer
        block_nums (List[int]): List of residual block numbers for each layer
        kernel_size (int): Kernel size of convolution layers in residual blocks
        stride (List[int]): List of stride values for each layer's first residual block

    Returns:
        paddle.Tensor: Output tensor from the third residual layer of the ResNet
    """

    def __init__(
        self,
        num_filter: int,
        map_num: List[int],
        block_nums: List[int],
        kernel_size: int,
        stride: List[int],
    ):
        super(ResnetStraight, self).__init__()
        self.in_channels = num_filter * map_num[0]
        self.stride = stride
        self.block_nums = block_nums
        self.kernel_size = kernel_size

        for layer_idx, (map_num_val, block_num, stride_val) in enumerate(
            zip(map_num[:3], block_nums[:3], stride[:3])
        ):
            layer = self.blocklayer(
                num_filter * map_num_val,
                block_num,
                kernel_size=self.kernel_size,
                stride=stride_val,
            )
            setattr(self, f"layer{layer_idx + 1}", layer)

    def blocklayer(
        self, out_channels: int, block_nums: int, kernel_size: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(
                    self.in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                ),
                nn.BatchNorm2D(out_channels),
            )

        layers = []
        for i in range(block_nums):
            layers.append(
                ResidualBlockWithDilation(
                    in_channels=self.in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1,
                    downsample=downsample if i == 0 else None,
                    is_top=True if i == 0 else False,
                )
            )
        self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3


class UVDocNet(BatchNormHFStateDictMixin, PretrainedModel):
    """
    UVDocNet: Image rectification network based on ResNet and multi-scale dilated convolution

    Args:
        config (UVDocNetConfig): Configuration object containing network hyperparameters

    Returns:
        List: List containing the transformed document image tensor (converted to numpy array on CPU)
    """

    config_class = UVDocNetConfig

    def __init__(self, config: UVDocNetConfig):
        super(UVDocNet, self).__init__(config)

        self.num_filter = config.num_filter
        self.in_channels = config.in_channels
        self.kernel_size = config.kernel_size
        self.stride = config.stride
        self.map_num = config.map_num
        self.block_nums = config.block_nums
        self.dilation_values = config.dilation_values
        self.padding_mode = config.padding_mode
        self.upsample_size = config.upsample_size
        self.upsample_mode = config.upsample_mode

        self.resnet_head = nn.Sequential(
            nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.num_filter * self.map_num[0],
                bias_attr=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2D(self.num_filter * self.map_num[0]),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=self.num_filter * self.map_num[0],
                out_channels=self.num_filter * self.map_num[0],
                bias_attr=False,
                kernel_size=self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm2D(self.num_filter * self.map_num[0]),
            nn.ReLU(),
        )

        self.resnet_down = ResnetStraight(
            self.num_filter,
            self.map_num,
            block_nums=self.block_nums,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )

        bridge_in_channels = self.num_filter * self.map_num[2]

        def _build_bridge(bridge_key: str) -> nn.Sequential:
            dilation = self.dilation_values[bridge_key]
            if isinstance(dilation, int):
                return nn.Sequential(
                    dilated_conv_bn_act(
                        bridge_in_channels, bridge_in_channels, dilation=dilation
                    )
                )
            else:
                return nn.Sequential(
                    *[
                        dilated_conv_bn_act(
                            bridge_in_channels, bridge_in_channels, dilation=d
                        )
                        for d in dilation
                    ]
                )

        self.bridge_1 = _build_bridge("bridge_1")
        self.bridge_2 = _build_bridge("bridge_2")
        self.bridge_3 = _build_bridge("bridge_3")
        self.bridge_4 = _build_bridge("bridge_4")
        self.bridge_5 = _build_bridge("bridge_5")
        self.bridge_6 = _build_bridge("bridge_6")

        self.bridge_concat = nn.Sequential(
            nn.Conv2D(
                in_channels=self.num_filter * self.map_num[2] * 6,
                out_channels=self.num_filter * self.map_num[2],
                bias_attr=False,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2D(self.num_filter * self.map_num[2]),
            nn.ReLU(),
        )

        self.out_point_positions2D = nn.Sequential(
            nn.Conv2D(
                in_channels=self.num_filter * self.map_num[2],
                out_channels=self.num_filter * self.map_num[0],
                bias_attr=False,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
            ),
            nn.BatchNorm2D(self.num_filter * self.map_num[0]),
            nn.PReLU(),
            nn.Conv2D(
                in_channels=self.num_filter * self.map_num[0],
                out_channels=2,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                padding_mode=self.padding_mode,
            ),
        )

    add_inference_operations("uvdoc_forward")

    @benchmark.timeit_with_options(name="uvdoc_forward")
    def forward(self, x: Any) -> List[paddle.Tensor]:
        x = paddle.to_tensor(x[0])

        image = x
        h_ori, w_ori = x.shape[2:]
        x = F.upsample(
            x,
            size=(self.upsample_size[0], self.upsample_size[1]),
            mode=self.upsample_mode,
            align_corners=True,
        )
        resnet_head = self.resnet_head(x)
        resnet_down = self.resnet_down(resnet_head)

        bridge_1 = self.bridge_1(resnet_down)
        bridge_2 = self.bridge_2(resnet_down)
        bridge_3 = self.bridge_3(resnet_down)
        bridge_4 = self.bridge_4(resnet_down)
        bridge_5 = self.bridge_5(resnet_down)
        bridge_6 = self.bridge_6(resnet_down)

        bridge_concat = paddle.concat(
            x=[bridge_1, bridge_2, bridge_3, bridge_4, bridge_5, bridge_6], axis=1
        )
        bridge = self.bridge_concat(bridge_concat)
        out_point_positions2D = self.out_point_positions2D(bridge)

        bm_up = F.upsample(
            out_point_positions2D,
            size=(h_ori, w_ori),
            mode=self.upsample_mode,
            align_corners=True,
        )
        bm = bm_up.transpose([0, 2, 3, 1])
        out = F.grid_sample(image, bm, align_corners=True)

        return [out.cpu().numpy()]

    def _get_forward_key_rules(self):
        default_rules = super()._get_forward_key_rules()
        custom_rules = [("out_point_positions2D.2._weight", "_weight", "weight")]
        return default_rules + custom_rules

    def _get_reverse_key_rules(self):
        default_rules = super()._get_reverse_key_rules()
        custom_rules = [("out_point_positions2D.2.weight", "weight", "_weight")]
        return default_rules + custom_rules
