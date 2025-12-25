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

from typing import Any, Dict, List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, KaimingNormal
from paddle.regularizer import L2Decay

from ....utils.benchmark import add_inference_operations, benchmark
from ...common.transformers.transformers import (
    BatchNormHFStateDictMixin,
    PretrainedModel,
)
from ._config_pp_ocrv5_server import PPOCRV5ServerDetConfig
from .pp_ocrv5_modules import DBHead, LearnableAffineBlock

kaiming_normal_ = KaimingNormal()
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


class ConvBNAct(nn.Layer):
    """
    ConvBNAct: Convolution + Batch Normalization + Activation (optional) with Learnable Affine Block

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Convolution kernel size, default is 3
        stride (int, optional): Convolution stride, default is 1
        padding (Union[int, str], optional): Padding value or mode (e.g. 'same'), default is 1
        groups (int, optional): Number of grouped convolution groups, default is 1
        use_act (bool, optional): Whether to apply ReLU activation, default is True
        use_lab (bool, optional): Whether to use LearnableAffineBlock after activation, default is False
        lr_mult (float, optional): Learning rate multiplier for conv/bn parameters, default is 1.0

    Returns:
        paddle.Tensor: Output tensor after convolution, BN and optional activation/affine transform
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Union[int, str] = 1,
        groups: int = 1,
        use_act: bool = True,
        use_lab: bool = False,
        lr_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        padding_val = padding if isinstance(padding, str) else (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding_val,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
        )
        if self.use_act:
            self.act = nn.ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock(lr_mult=lr_mult)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(nn.Layer):
    """
    LightConvBNAct: Lightweight depthwise separable convolution block with BN and activation

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of depthwise convolution
        use_lab (bool, optional): Whether to use LearnableAffineBlock in ConvBNAct, default is False
        lr_mult (float, optional): Learning rate multiplier for conv/bn parameters, default is 1.0
        **kwargs: Additional keyword arguments

    Returns:
        paddle.Tensor: Output tensor after pointwise conv (no act) + depthwise conv (with act)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_lab: bool = False,
        lr_mult: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Layer):
    """
    StemBlock: Multi-stage convolution stem block with pooling and concatenation

    Args:
        in_channels (int): Number of input channels
        mid_channels (int): Number of intermediate channels for stem layers
        out_channels (int): Number of output channels
        use_lab (bool, optional): Whether to use LearnableAffineBlock in ConvBNAct, default is False
        lr_mult (float, optional): Learning rate multiplier for conv/bn parameters, default is 1.0

    Returns:
        paddle.Tensor: Output tensor after multi-stage convolution, pooling and concatenation
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_lab: bool = False,
        lr_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=1, ceil_mode=True, padding="SAME"
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = paddle.concat([x1, x2], 1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


class HGV2_Block(nn.Layer):
    """
    HGV2_Block: Multi-layer convolution block with feature aggregation and residual connection

    Args:
        in_channels (int): Number of input channels
        mid_channels (int): Number of intermediate channels for each layer
        out_channels (int): Number of output channels after aggregation
        kernel_size (int, optional): Kernel size of convolution layers, default is 3
        layer_num (int, optional): Number of convolution layers in the block, default is 6
        identity (bool, optional): Whether to add identity residual connection, default is False
        light_block (bool, optional): Whether to use LightConvBNAct (True) or ConvBNAct (False), default is True
        use_lab (bool, optional): Whether to use LearnableAffineBlock in conv layers, default is False
        lr_mult (float, optional): Learning rate multiplier for conv/bn parameters, default is 1.0

    Returns:
        paddle.Tensor: Output tensor after multi-layer conv, feature concatenation and aggregation
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        layer_num: int = 6,
        identity: bool = False,
        light_block: bool = True,
        use_lab: bool = False,
        lr_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.identity = identity

        self.layers = nn.LayerList()
        block_type = "LightConvBNAct" if light_block else "ConvBNAct"
        for i in range(layer_num):
            self.layers.append(
                eval(block_type)(
                    in_channels=in_channels if i == 0 else mid_channels,
                    out_channels=mid_channels,
                    stride=1,
                    kernel_size=kernel_size,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = paddle.concat(output, axis=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x += identity
        return x


class HGV2_Stage(nn.Layer):
    """
    HGV2_Stage: Sequential HGV2_Block layers with optional depthwise downsampling

    Args:
        in_channels (int): Number of input channels
        mid_channels (int): Number of intermediate channels for HGV2_Block layers
        out_channels (int): Number of output channels for each HGV2_Block
        block_num (int): Number of HGV2_Block in the stage
        layer_num (int, optional): Number of convolution layers in each HGV2_Block, default is 6
        is_downsample (bool, optional): Whether to apply depthwise downsampling at stage start, default is True
        light_block (bool, optional): Whether to use LightConvBNAct in HGV2_Block, default is True
        kernel_size (int, optional): Kernel size of convolution layers in HGV2_Block, default is 3
        use_lab (bool, optional): Whether to use LearnableAffineBlock in conv layers, default is False
        stride (int, optional): Stride for downsampling convolution, default is 2
        lr_mult (float, optional): Learning rate multiplier for conv/bn parameters, default is 1.0

    Returns:
        paddle.Tensor: Output tensor after optional downsampling and sequential HGV2_Block processing
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        block_num: int,
        layer_num: int = 6,
        is_downsample: bool = True,
        light_block: bool = True,
        kernel_size: int = 3,
        use_lab: bool = False,
        stride: int = 2,
        lr_mult: float = 1.0,
    ) -> None:

        super().__init__()
        self.is_downsample = is_downsample
        if self.is_downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult,
            )

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HGV2_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.is_downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


class PPHGNetV2(nn.Layer):
    """
    PPHGNetV2: Hierarchical feature extraction network with stem block and multi-stage HGV2 blocks

    Args:
        stage_config (Dict[str, Tuple]): Dictionary of stage configurations, each tuple contains (in_channels, mid_channels, out_channels, block_num, is_downsample, light_block, kernel_size, layer_num, stride)
        stem_channels (Tuple[int, int, int]): Stem block channels (in, mid, out)
        use_lab (bool): Whether to use LearnableAffineBlock in ConvBNAct layers
        use_last_conv (bool): Whether to use last convolution layer (unused in current implementation)
        class_expand (float): Expansion factor for classification head channels
        class_num (int): Number of classification classes
        lr_mult_list (List[float]): Learning rate multipliers for stem and each stage
        det (bool): Whether the network is used for detection (controls output indices)
        out_indices (List[int]): Indices of stages to output features for detection
        **kwargs: Additional keyword arguments

    Returns:
        List[paddle.Tensor]: List of feature tensors from specified stages (only when det=True)
    """

    def __init__(
        self,
        stage_config: Dict[str, Tuple],
        stem_channels: Tuple[int, int, int],
        use_lab: bool,
        use_last_conv: bool,
        class_expand: float,
        class_num: int,
        lr_mult_list: List[float],
        det: bool,
        out_indices: List[int],
        **kwargs,
    ) -> None:
        super().__init__()
        self.det = det
        self.use_lab = use_lab
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.class_num = class_num
        self.out_indices = out_indices
        self.out_channels = []

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab,
            lr_mult=lr_mult_list[0],
        )

        # stages
        self.stages = nn.LayerList()
        for i, k in enumerate(stage_config):
            (
                in_channels,
                mid_channels,
                out_channels,
                block_num,
                is_downsample,
                light_block,
                kernel_size,
                layer_num,
                stride,
            ) = stage_config[k]
            self.stages.append(
                HGV2_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    is_downsample,
                    light_block,
                    kernel_size,
                    use_lab,
                    stride,
                    lr_mult=lr_mult_list[i + 1],
                )
            )
            if i in self.out_indices:
                self.out_channels.append(out_channels)

        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2D)):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                zeros_(m.bias)

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        x = self.stem(x)
        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.det and i in self.out_indices:
                out.append(x)
        return out


class DSConv(nn.Layer):
    """
    DSConv: Depthwise separable convolution with bottleneck and residual connection

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of depthwise convolution
        padding (Union[int, str]): Padding value or mode (e.g. 'SAME') for depthwise conv
        stride (int, optional): Stride for depthwise convolution, default is 1
        groups (Optional[int], optional): Number of groups for depthwise conv (default: in_channels)
        if_act (bool, optional): Whether to apply activation after second BN, default is True
        act (str, optional): Activation function type ('relu' or 'hardswish'), default is 'relu'
        **kwargs: Additional keyword arguments

    Returns:
        paddle.Tensor: Output tensor after depthwise separable convolution and optional residual connection
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: Union[int, str],
        stride: int = 1,
        groups: Optional[int] = None,
        if_act: bool = True,
        act: str = "relu",
        **kwargs,
    ) -> None:
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )

        self.bn1 = nn.BatchNorm(num_channels=in_channels, act=None)

        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias_attr=False,
        )

        self.bn2 = nn.BatchNorm(num_channels=int(in_channels * 4), act=None)

        self.conv3 = nn.Conv2D(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias_attr=False,
        )
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False,
            )

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class IntraCLBlock(nn.Layer):
    """
    IntraCLBlock: Multi-scale convolution block with vertical/horizontal kernel fusion

    Args:
        in_channels (int): Number of input channels
        reduce_factor (int): Channel reduction ratio for 1x1 convolution
        intraclblock_config (dict): Configuration dict for convolution layers, includes:
            - reduce_channel: (kernel_size, stride, padding) for channel reduction 1x1 conv
            - return_channel: (kernel_size, stride, padding) for channel recovery 1x1 conv
            - v_layer_7x1/5x1/3x1: (kernel_size, stride, padding) for vertical (Hx1) conv
            - q_layer_1x7/1x5/1x3: (kernel_size, stride, padding) for horizontal (1xW) conv
            - c_layer_7x7/5x5/3x3: (kernel_size, stride, padding) for cross (HxW) conv

    Returns:
        paddle.Tensor: Output tensor after multi-scale conv fusion and residual connection
    """

    def __init__(
        self, in_channels: int, reduce_factor: int, intraclblock_config: dict
    ) -> None:
        super(IntraCLBlock, self).__init__()

        self.channels = in_channels
        self.reduce_factor = reduce_factor
        self.intraclblock_config = intraclblock_config

        reduced_ch = self.channels // self.reduce_factor

        self.conv1x1_reduce_channel = nn.Conv2d(
            self.channels, reduced_ch, *self.intraclblock_config["reduce_channel"]
        )
        self.conv1x1_return_channel = nn.Conv2d(
            reduced_ch, self.channels, *self.intraclblock_config["return_channel"]
        )

        self.v_layer_7x1 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["v_layer_7x1"]
        )
        self.v_layer_5x1 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["v_layer_5x1"]
        )
        self.v_layer_3x1 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["v_layer_3x1"]
        )

        self.q_layer_1x7 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["q_layer_1x7"]
        )
        self.q_layer_1x5 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["q_layer_1x5"]
        )
        self.q_layer_1x3 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["q_layer_1x3"]
        )

        self.c_layer_7x7 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["c_layer_7x7"]
        )
        self.c_layer_5x5 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["c_layer_5x5"]
        )
        self.c_layer_3x3 = nn.Conv2d(
            reduced_ch, reduced_ch, *self.intraclblock_config["c_layer_3x3"]
        )

        self.bn = nn.BatchNorm2D(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x_new = self.conv1x1_reduce_channel(x)

        x_7 = (
            self.c_layer_7x7(x_new) + self.v_layer_7x1(x_new) + self.q_layer_1x7(x_new)
        )
        x_5 = self.c_layer_5x5(x_7) + self.v_layer_5x1(x_7) + self.q_layer_1x5(x_7)
        x_3 = self.c_layer_3x3(x_5) + self.v_layer_3x1(x_5) + self.q_layer_1x3(x_5)

        x_relation = self.conv1x1_return_channel(x_3)
        x_relation = self.bn(x_relation)
        x_relation = self.relu(x_relation)

        return x + x_relation


class LKPAN(nn.Layer):
    """
    LKPAN: Feature pyramid network with multi-scale aggregation and IntraCL enhancement

    Args:
        in_channels (List[int]): List of input channel numbers for multi-scale feature maps
        out_channels (int): Number of output channels for 1x1 convolution layers
        mode (str): Network mode ('lite' for DSConv, 'large' for standard Conv2D)
        reduce_factor (int): Channel reduction ratio for IntraCLBlock modules
        intraclblock_config (dict): Configuration dict for convolution layers, includes:
            - reduce_channel: (kernel_size, stride, padding) for channel reduction 1x1 conv
            - return_channel: (kernel_size, stride, padding) for channel recovery 1x1 conv
            - v_layer_7x1/5x1/3x1: (kernel_size, stride, padding) for vertical (Hx1) conv
            - q_layer_1x7/1x5/1x3: (kernel_size, stride, padding) for horizontal (1xW) conv
            - c_layer_7x7/5x5/3x3: (kernel_size, stride, padding) for cross (HxW) conv
        upsample_mode (str): Interpolation mode for upsample operation
        upsample_align_mode (int): Align mode for upsample operation
        **kwargs: Additional keyword arguments

    Returns:
        paddle.Tensor: Fused feature tensor after multi-scale feature aggregation, IntraCLBlock enhancement and concatenation
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        mode: str,
        reduce_factor: int,
        intraclblock_config: dict,
        upsample_mode: str,
        upsample_align_mode: int,
        **kwargs,
    ) -> None:
        super(LKPAN, self).__init__()
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.upsample_align_mode = upsample_align_mode
        weight_attr = nn.initializer.KaimingUniform()

        if mode.lower() == "lite":
            p_layer = DSConv
        elif mode.lower() == "large":
            p_layer = nn.Conv2D
        else:
            raise ValueError(
                "mode can only be one of ['lite', 'large'], but received {}".format(
                    mode
                )
            )

        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()
        self.pan_head_conv = nn.LayerList()
        self.pan_lat_conv = nn.LayerList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                nn.Conv2D(
                    in_channels=in_channels[i],
                    out_channels=self.out_channels,
                    kernel_size=1,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            self.inp_conv.append(
                p_layer(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

            if i > 0:
                self.pan_head_conv.append(
                    nn.Conv2D(
                        in_channels=self.out_channels // 4,
                        out_channels=self.out_channels // 4,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        weight_attr=ParamAttr(initializer=weight_attr),
                        bias_attr=False,
                    )
                )
            self.pan_lat_conv.append(
                p_layer(
                    in_channels=self.out_channels // 4,
                    out_channels=self.out_channels // 4,
                    kernel_size=9,
                    padding=4,
                    weight_attr=ParamAttr(initializer=weight_attr),
                    bias_attr=False,
                )
            )

        self.incl1 = IntraCLBlock(
            self.out_channels // 4,
            reduce_factor=reduce_factor,
            intraclblock_config=intraclblock_config,
        )
        self.incl2 = IntraCLBlock(
            self.out_channels // 4,
            reduce_factor=reduce_factor,
            intraclblock_config=intraclblock_config,
        )
        self.incl3 = IntraCLBlock(
            self.out_channels // 4,
            reduce_factor=reduce_factor,
            intraclblock_config=intraclblock_config,
        )
        self.incl4 = IntraCLBlock(
            self.out_channels // 4,
            reduce_factor=reduce_factor,
            intraclblock_config=intraclblock_config,
        )

    def forward(self, x: List[paddle.Tensor]) -> paddle.Tensor:
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.upsample(
            in5,
            scale_factor=2,
            mode=self.upsample_mode,
            align_mode=self.upsample_align_mode,
        )
        out3 = in3 + F.upsample(
            out4,
            scale_factor=2,
            mode=self.upsample_mode,
            align_mode=self.upsample_align_mode,
        )
        out2 = in2 + F.upsample(
            out3,
            scale_factor=2,
            mode=self.upsample_mode,
            align_mode=self.upsample_align_mode,
        )

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        p5 = self.incl4(p5)
        p4 = self.incl3(p4)
        p3 = self.incl2(p3)
        p2 = self.incl1(p2)

        p5 = F.upsample(
            p5,
            scale_factor=8,
            mode=self.upsample_mode,
            align_mode=self.upsample_align_mode,
        )
        p4 = F.upsample(
            p4,
            scale_factor=4,
            mode=self.upsample_mode,
            align_mode=self.upsample_align_mode,
        )
        p3 = F.upsample(
            p3,
            scale_factor=2,
            mode=self.upsample_mode,
            align_mode=self.upsample_align_mode,
        )

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse


class ConvBNLayer(nn.Layer):
    """
    ConvBNLayer: Basic convolution + batch normalization + optional activation block

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of convolution layer
        stride (int): Convolution stride
        padding (Union[int, str]): Padding value or mode (e.g. 'SAME') for convolution
        groups (int, optional): Number of grouped convolution groups, default is 1
        if_act (bool, optional): Whether to apply activation function, default is True
        act (Optional[str], optional): Activation function type ('relu' or 'hardswish'), default is None

    Returns:
        paddle.Tensor: Output tensor after convolution, BN and optional activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: Union[int, str],
        groups: int = 1,
        if_act: bool = True,
        act: Optional[str] = None,
    ) -> None:
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False,
        )

        self.bn = nn.BatchNorm(num_channels=out_channels, act=None)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print(
                    "The activation function({}) is selected incorrectly.".format(
                        self.act
                    )
                )
                exit()
        return x


class LocalModule(nn.Layer):
    """
    LocalModule: Feature enhancement module with concatenation and 1x1 projection

    Args:
        in_c (int): Number of input channels (before concatenation)
        mid_c (int): Number of intermediate channels for 3x3 ConvBNLayer
        act (str): Activation function type for ConvBNLayer

    Returns:
        paddle.Tensor: 1-channel output tensor after concatenation, conv and projection
    """

    def __init__(self, in_c: int, mid_c: int, act: str) -> None:
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act=act)
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def forward(self, x: paddle.Tensor, init_map: paddle.Tensor) -> paddle.Tensor:
        outf = paddle.concat([init_map, x], axis=1)
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):
    """
    PFHeadLocal: Enhanced DB head with local feature refinement for detection

    Args:
        in_channels (int): Number of input channels
        k (int): DB head hyperparameter (kernel factor)
        mode (str): Module size mode ('large' or 'small') to control intermediate channels
        scale_factor (int): Upsampling scale factor for feature maps
        act (str): Activation function type for LocalModule
        upsample_mode (str): Interpolation mode for upsample operation
        upsample_align_mode (int): Align mode for upsample operation
        **kwargs: Additional keyword arguments for parent DBHead class

    Returns:
        paddle.Tensor: Fused binarization map (average of base map and enhanced local map)
    """

    def __init__(
        self,
        in_channels: int,
        k: int,
        mode: str,
        scale_factor: int,
        act: str,
        upsample_mode: str,
        upsample_align_mode: int,
        **kwargs: Any,
    ) -> None:
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode

        self.up_conv = nn.Upsample(
            scale_factor=scale_factor,
            mode=upsample_mode,
            align_mode=upsample_align_mode,
        )

        if mode == "large":
            mid_ch = in_channels // 4
        elif mode == "small":
            mid_ch = in_channels // 8
        else:
            raise ValueError(f"mode must be 'large' or 'small', currently {mode}")
        self.cbn_layer = LocalModule(in_channels // 4, mid_ch, act)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        base_maps, f = self.binarize(x, return_f=True)

        cbn_maps = self.cbn_layer(self.up_conv(f), base_maps)
        cbn_maps = F.sigmoid(cbn_maps)

        return 0.5 * (base_maps + cbn_maps)


class PPOCRV5ServerDet(BatchNormHFStateDictMixin, PretrainedModel):
    """
    PPOCRV5ServerDet: Server-side OCR detection model with PPHGNetV2, LKPAN and PFHeadLocal

    Args:
        config (PPOCRV5ServerDetConfig): Configuration object containing model hyperparameters

    Returns:
        List: List containing the detection output tensor (converted to numpy array on CPU)
    """

    config_class = PPOCRV5ServerDetConfig

    def __init__(self, config: PPOCRV5ServerDetConfig) -> None:
        super().__init__(config)

        self.upsample_mode = config.upsample_mode
        self.upsample_align_mode = config.upsample_align_mode
        self.backbone_stem_channels = config.backbone_stem_channels
        self.backbone_stage_config = config.backbone_stage_config
        self.backbone_use_lab = config.backbone_use_lab
        self.backbone_use_last_conv = config.backbone_use_last_conv
        self.backbone_class_expand = config.backbone_class_expand
        self.backbone_class_num = config.backbone_class_num
        self.backbone_lr_mult_list = config.backbone_lr_mult_list
        self.backbone_det = config.backbone_det
        self.backbone_out_indices = config.backbone_out_indices

        self.neck_out_channels = config.neck_out_channels
        self.neck_mode = config.neck_mode
        self.neck_reduce_factor = config.neck_reduce_factor
        self.neck_intraclblock_config = config.neck_intraclblock_config

        self.head_in_channels = config.head_in_channels
        self.head_k = config.head_k
        self.head_mode = config.head_mode
        self.head_scale_factor = config.head_scale_factor
        self.head_act = config.head_act
        self.head_kernel_list = config.head_kernel_list
        self.head_fix_nan = config.head_fix_nan

        self.backbone = PPHGNetV2(
            stage_config=self.backbone_stage_config,
            stem_channels=self.backbone_stem_channels,
            use_lab=self.backbone_use_lab,
            use_last_conv=self.backbone_use_last_conv,
            class_expand=self.backbone_class_expand,
            class_num=self.backbone_class_num,
            lr_mult_list=self.backbone_lr_mult_list,
            det=self.backbone_det,
            out_indices=self.backbone_out_indices,
        )

        neck_in_channels = self.backbone.out_channels
        self.neck = LKPAN(
            in_channels=neck_in_channels,
            out_channels=self.neck_out_channels,
            mode=self.neck_mode,
            reduce_factor=self.neck_reduce_factor,
            intraclblock_config=self.neck_intraclblock_config,
            upsample_mode=self.upsample_mode,
            upsample_align_mode=self.upsample_align_mode,
        )

        head_in_channels = self.neck.out_channels
        self.head = PFHeadLocal(
            in_channels=head_in_channels,
            k=self.head_k,
            mode=self.head_mode,
            scale_factor=self.head_scale_factor,
            act=self.head_act,
            upsample_mode=self.upsample_mode,
            upsample_align_mode=self.upsample_align_mode,
            kernel_list=self.head_kernel_list,
            fix_nan=self.head_fix_nan,
        )

    add_inference_operations("pp_ocrv5_server_det_forward")

    @benchmark.timeit_with_options(name="pp_ocrv5_server_det_forward")
    def forward(self, x: List) -> List:

        x = paddle.to_tensor(x[0])

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return [x.cpu().numpy()]
