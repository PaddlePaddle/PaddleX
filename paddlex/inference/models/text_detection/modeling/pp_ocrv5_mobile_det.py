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

from typing import Any, Dict, List, Optional, Union

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
from ._config_pp_ocrv5_mobile import PPOCRV5MobileDetConfig
from .pp_ocrv5_modules import DBHead, LearnableAffineBlock


def make_divisible(
    v: Union[int, float], divisor: int = 16, min_value: Optional[int] = None
) -> int:
    """
    make_divisible: Adjust channel number to be divisible by specified divisor (network width optimization)

    Args:
        v (Union[int, float]): Original channel number
        divisor (int, optional): Divisor for channel adjustment, default 16
        min_value (Optional[int], optional): Minimum channel number after adjustment, default None

    Returns:
        int: Adjusted channel number (integer)
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Act(nn.Layer):
    """
    Act: Activation layer with Learnable Affine Block (LAB)

    Args:
        act (str): Activation type, "relu" or "hswish"
        lr_mult (float): Learning rate multiplier for LAB alpha
        lab_lr (float): Learning rate multiplier for LAB beta

    Returns:
        paddle.Tensor: Output tensor after activation and LAB
    """

    def __init__(self, act: str, lr_mult: float, lab_lr: float):
        super().__init__()
        if act == "hswish":
            self.act = nn.Hardswish()
        else:
            assert act == "relu"
            self.act = nn.ReLU()
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return self.lab(self.act(x))


class ConvBNLayer(nn.Layer):
    """
    ConvBNLayer: Convolution + Batch Normalization combination layer

    Args:
        in_channels (int): Input channel number
        out_channels (int): Output channel number
        kernel_size (int): Convolution kernel size
        stride (int): Convolution stride
        lr_mult (float): Learning rate multiplier for conv/BN params
        groups (int, optional): Group convolution number, default 1

    Returns:
        paddle.Tensor: Output tensor after conv and BN
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        lr_mult: float,
        groups: int = 1,
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

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


class LearnableRepLayer(nn.Layer):
    """
    LearnableRepLayer: Learnable representation layer with multi-branch convolution fusion

    Args:
        in_channels (int): Input channel number
        out_channels (int): Output channel number
        kernel_size (int): Convolution kernel size
        act (str): Activation type, "relu" or "hswish"
        stride (int): Convolution stride
        lr_mult (float): Learning rate multiplier for conv/BN/LAB params
        lab_lr (float): Learning rate multiplier for LAB beta
        num_conv_branches (int): Number of kxk conv branches
        groups (int, optional): Group convolution number, default 1

    Returns:
        paddle.Tensor: Output tensor after rep branches fusion and activation
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        act: str,
        stride: int,
        lr_mult: float,
        lab_lr: float,
        num_conv_branches: int,
        groups: int = 1,
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
        self.act = Act(act=act, lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:

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
    """
    SELayer: Squeeze-and-Excitation channel attention layer

    Args:
        channel (int): Input/output channel number
        reduction (int): Channel reduction ratio for excitation
        lr_mult (float): Learning rate multiplier for conv params

    Returns:
        paddle.Tensor: Output tensor after channel attention weighting
    """

    def __init__(self, channel: int, reduction: int, lr_mult: float):
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

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class LCNetV3Block(nn.Layer):
    """
    LCNetV3Block: Depthwise separable convolution block with SE attention (LCNetV3)

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        act (str): Activation function type
        stride (int): Convolution stride for depthwise conv
        dw_size (int): Kernel size of depthwise convolution
        use_se (bool): Whether to use SE channel attention module
        conv_kxk_num (int): Number of conv branches in LearnableRepLayer
        reduction (int): Channel reduction ratio for SE module
        lr_mult (float): Learning rate multiplier for convolution parameters
        lab_lr (float): Learning rate multiplier for learnable representation layer

    Returns:
        paddle.Tensor: Output tensor after depthwise conv, SE (optional) and pointwise conv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act: str,
        stride: int,
        dw_size: int,
        use_se: bool,
        conv_kxk_num: int,
        reduction: int,
        lr_mult: float,
        lab_lr: float,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            act=act,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )
        if use_se:
            self.se = SELayer(in_channels, reduction=reduction, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act=act,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPLCNetV3(nn.Layer):
    """
    PPLCNetV3: Lightweight convolutional network with learnable representation layers

    Args:
        scale (float): Channel scale factor for network width adjustment
        conv_kxk_num (int): Number of conv branches in LearnableRepLayer of LCNetV3Block
        reduction (int): Channel reduction ratio for SE module in LCNetV3Block
        act (str): Activation function type used in LCNetV3Block
        lr_mult_list (List[float]): Learning rate multipliers for different layers, length must be 6
        lab_lr (float): Learning rate multiplier for learnable representation layer in LCNetV3Block
        net_config (Dict[str, Any]): Network configuration dict containing block parameters and output channels
        out_channels (int): Base number of output channels before scale adjustment
        **kwargs: Additional keyword arguments

    Returns:
        List[paddle.Tensor]: List of 4 feature tensors from different stages after 1x1 conv projection
    """

    def __init__(
        self,
        scale: float,
        conv_kxk_num: int,
        reduction: int,
        act: str,
        lr_mult_list: List[float],
        lab_lr: float,
        net_config: Dict[str, Any],
        out_channels: int,
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

        def _build_blocks(block_key, lr_mult_idx):
            return nn.Sequential(
                *[
                    LCNetV3Block(
                        in_channels=make_divisible(in_c * scale),
                        out_channels=make_divisible(out_c * scale),
                        act=act,
                        dw_size=k,
                        stride=s,
                        use_se=se,
                        conv_kxk_num=conv_kxk_num,
                        reduction=reduction,
                        lr_mult=self.lr_mult_list[lr_mult_idx],
                        lab_lr=lab_lr,
                    )
                    for i, (k, in_c, out_c, s, se) in enumerate(
                        self.net_config[block_key]
                    )
                ]
            )

        self.blocks2 = _build_blocks("blocks2", 1)
        self.blocks3 = _build_blocks("blocks3", 2)
        self.blocks4 = _build_blocks("blocks4", 3)
        self.blocks5 = _build_blocks("blocks5", 4)
        self.blocks6 = _build_blocks("blocks6", 5)

        mv_c = self.net_config["layer_list_out_channels"]

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

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
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
    """
    SEModule: Simplified Squeeze-and-Excitation channel attention module

    Args:
        in_channels (int): Number of input channels
        reduction (int): Channel reduction ratio for excitation layer

    Returns:
        paddle.Tensor: Output tensor after channel attention weighting
    """

    def __init__(self, in_channels: int, reduction: int):
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

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs


class RSELayer(nn.Layer):
    """
    RSELayer: Residual SE layer with convolution and shortcut connection

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size of convolution layer
        shortcut (bool): Whether to add shortcut connection (residual) with SE output
        reduction (int): Channel reduction ratio for SE module

    Returns:
        paddle.Tensor: Output tensor after convolution, SE attention and optional shortcut
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        shortcut: bool,
        reduction: int,
    ):
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
        self.se_block = SEModule(self.out_channels, reduction=reduction)
        self.shortcut = shortcut

    def forward(self, ins: paddle.Tensor) -> paddle.Tensor:
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Layer):
    """
    RSEFPN: Feature Pyramid Network with Residual SE attention

    Args:
        in_channels (List[int]): List of input channel numbers for multi-scale feature maps
        out_channels (int): Number of output channels for RSELayer convolution
        shortcut (bool): Whether to use shortcut connection in RSELayer
        reduction (int): Channel reduction ratio for SE module in RSELayer
        **kwargs: Additional keyword arguments

    Returns:
        paddle.Tensor: Fused feature tensor after multi-scale feature aggregation and concatenation
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        shortcut: bool,
        reduction: int,
        **kwargs,
    ):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.LayerList()
        self.inp_conv = nn.LayerList()

        for i in range(len(in_channels)):
            self.ins_conv.append(
                RSELayer(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    shortcut=shortcut,
                    reduction=reduction,
                )
            )
            self.inp_conv.append(
                RSELayer(
                    out_channels,
                    out_channels // 4,
                    kernel_size=3,
                    shortcut=shortcut,
                    reduction=reduction,
                )
            )

    def forward(self, x: List[paddle.Tensor]) -> paddle.Tensor:
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
    """
    PPOCRV5MobileDet: Lightweight OCR detection model based on PPLCNetV3, RSEFPN and DBHead

    Args:
        config (PPOCRV5MobileDetConfig): Configuration object containing model hyperparameters

    Returns:
        List: List containing the detection output tensor (converted to numpy array on CPU)
    """

    config_class = PPOCRV5MobileDetConfig

    def __init__(self, config: PPOCRV5MobileDetConfig):
        super().__init__(config)

        self.backbone_scale = config.backbone_scale
        self.backbone_det = config.backbone_det
        self.backbone_conv_kxk_num = config.backbone_conv_kxk_num
        self.backbone_reduction = config.backbone_reduction
        self.backbone_act = config.backbone_act
        self.backbone_lr_mult_list = config.backbone_lr_mult_list
        self.backbone_lab_lr = config.backbone_lab_lr
        self.backbone_net_config = config.backbone_net_config
        self.backbone_out_channels = config.backbone_out_channels

        self.neck_out_channels = config.neck_out_channels
        self.neck_shortcut = config.neck_shortcut

        self.head_k = config.head_k
        self.head_kernel_list = config.head_kernel_list
        self.head_fix_nan = config.head_fix_nan

        self.backbone = PPLCNetV3(
            scale=self.backbone_scale,
            conv_kxk_num=self.backbone_conv_kxk_num,
            reduction=self.backbone_reduction,
            act=self.backbone_act,
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
            reduction=self.backbone_reduction,
        )

        head_in_channels = self.neck_out_channels
        self.head = DBHead(
            in_channels=head_in_channels,
            k=self.head_k,
            kernel_list=self.head_kernel_list,
            fix_nan=self.head_fix_nan,
        )

    add_inference_operations("pp_ocrv5_mobile_det_forward")

    @benchmark.timeit_with_options(name="pp_ocrv5_mobile_det_forward")
    def forward(self, x: List) -> List:

        if isinstance(x, (list, tuple)):
            x = x[0]
        if not isinstance(x, paddle.Tensor):
            x = paddle.to_tensor(x)

        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return [x.cpu().numpy()]
