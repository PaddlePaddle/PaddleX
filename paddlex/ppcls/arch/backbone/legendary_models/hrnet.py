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

import math
import paddle
from paddle import nn
from paddle import ParamAttr
from paddle.nn.functional import upsample
from paddle.nn.initializer import Uniform

from paddlex.ppcls.arch.backbone.base.theseus_layer import TheseusLayer, Identity
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "HRNet_W18_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W18_C_pretrained.pdparams",
    "HRNet_W30_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W30_C_pretrained.pdparams",
    "HRNet_W32_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W32_C_pretrained.pdparams",
    "HRNet_W40_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W40_C_pretrained.pdparams",
    "HRNet_W44_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W44_C_pretrained.pdparams",
    "HRNet_W48_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W48_C_pretrained.pdparams",
    "HRNet_W64_C":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/HRNet_W64_C_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act is None:
        return Identity()
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act="relu"):
        super().__init__()

        self.conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)
        self.bn = nn.BatchNorm(num_filters, act=None)
        self.act = _create_act(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BottleneckBlock(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False):
        super().__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu")
        self.conv3 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if self.downsample:
            self.conv_down = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                act=None)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16)
        self.relu = nn.ReLU()

    def forward(self, x, res_dict=None):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.downsample:
            residual = self.conv_down(residual)
        if self.has_se:
            x = self.se(x)
        x = paddle.add(x=residual, y=x)
        x = self.relu(x)
        return x


class BasicBlock(nn.Layer):
    def __init__(self, num_channels, num_filters, has_se=False):
        super().__init__()

        self.has_se = has_se

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act="relu")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act=None)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)

        if self.has_se:
            x = self.se(x)

        x = paddle.add(x=residual, y=x)
        x = self.relu(x)
        return x


class SELayer(TheseusLayer):
    def __init__(self, num_channels, num_filters, reduction_ratio):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.fc_squeeze = nn.Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
        self.relu = nn.ReLU()
        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.fc_excitation = nn.Linear(
            med_ch,
            num_filters,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, res_dict=None):
        residual = x
        x = self.avg_pool(x)
        x = paddle.squeeze(x, axis=[2, 3])
        x = self.fc_squeeze(x)
        x = self.relu(x)
        x = self.fc_excitation(x)
        x = self.sigmoid(x)
        x = paddle.unsqueeze(x, axis=[2, 3])
        x = residual * x
        return x


class Stage(TheseusLayer):
    def __init__(self, num_modules, num_filters, has_se=False):
        super().__init__()

        self._num_modules = num_modules

        self.stage_func_list = nn.LayerList()
        for i in range(num_modules):
            self.stage_func_list.append(
                HighResolutionModule(
                    num_filters=num_filters, has_se=has_se))

    def forward(self, x, res_dict=None):
        x = x
        for idx in range(self._num_modules):
            x = self.stage_func_list[idx](x)
        return x


class HighResolutionModule(TheseusLayer):
    def __init__(self, num_filters, has_se=False):
        super().__init__()

        self.basic_block_list = nn.LayerList()

        for i in range(len(num_filters)):
            self.basic_block_list.append(
                nn.Sequential(*[
                    BasicBlock(
                        num_channels=num_filters[i],
                        num_filters=num_filters[i],
                        has_se=has_se) for j in range(4)
                ]))

        self.fuse_func = FuseLayers(
            in_channels=num_filters, out_channels=num_filters)

    def forward(self, x, res_dict=None):
        out = []
        for idx, xi in enumerate(x):
            basic_block_list = self.basic_block_list[idx]
            for basic_block_func in basic_block_list:
                xi = basic_block_func(xi)
            out.append(xi)
        out = self.fuse_func(out)
        return out


class FuseLayers(TheseusLayer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self._actual_ch = len(in_channels)
        self._in_channels = in_channels

        self.residual_func_list = nn.LayerList()
        self.relu = nn.ReLU()
        for i in range(len(in_channels)):
            for j in range(len(in_channels)):
                if j > i:
                    self.residual_func_list.append(
                        ConvBNLayer(
                            num_channels=in_channels[j],
                            num_filters=out_channels[i],
                            filter_size=1,
                            stride=1,
                            act=None))
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            self.residual_func_list.append(
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[i],
                                    filter_size=3,
                                    stride=2,
                                    act=None))
                            pre_num_filters = out_channels[i]
                        else:
                            self.residual_func_list.append(
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[j],
                                    filter_size=3,
                                    stride=2,
                                    act="relu"))
                            pre_num_filters = out_channels[j]

    def forward(self, x, res_dict=None):
        out = []
        residual_func_idx = 0
        for i in range(len(self._in_channels)):
            residual = x[i]
            for j in range(len(self._in_channels)):
                if j > i:
                    xj = self.residual_func_list[residual_func_idx](x[j])
                    residual_func_idx += 1

                    xj = upsample(xj, scale_factor=2**(j - i), mode="nearest")
                    residual = paddle.add(x=residual, y=xj)
                elif j < i:
                    xj = x[j]
                    for k in range(i - j):
                        xj = self.residual_func_list[residual_func_idx](xj)
                        residual_func_idx += 1

                    residual = paddle.add(x=residual, y=xj)

            residual = self.relu(residual)
            out.append(residual)

        return out


class LastClsOut(TheseusLayer):
    def __init__(self,
                 num_channel_list,
                 has_se,
                 num_filters_list=[32, 64, 128, 256]):
        super().__init__()

        self.func_list = nn.LayerList()
        for idx in range(len(num_channel_list)):
            self.func_list.append(
                BottleneckBlock(
                    num_channels=num_channel_list[idx],
                    num_filters=num_filters_list[idx],
                    has_se=has_se,
                    downsample=True))

    def forward(self, x, res_dict=None):
        out = []
        for idx, xi in enumerate(x):
            xi = self.func_list[idx](xi)
            out.append(xi)
        return out


class HRNet(TheseusLayer):
    """
    HRNet
    Args:
        width: int=18. Base channel number of HRNet.
        has_se: bool=False. If 'True', add se module to HRNet.
        class_num: int=1000. Output num of last fc layer.
    Returns:
        model: nn.Layer. Specific HRNet model depends on args.
    """

    def __init__(self,
                 width=18,
                 has_se=False,
                 class_num=1000,
                 return_patterns=None):
        super().__init__()

        self.width = width
        self.has_se = has_se
        self._class_num = class_num

        channels_2 = [self.width, self.width * 2]
        channels_3 = [self.width, self.width * 2, self.width * 4]
        channels_4 = [
            self.width, self.width * 2, self.width * 4, self.width * 8
        ]

        self.conv_layer1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            stride=2,
            act="relu")

        self.conv_layer1_2 = ConvBNLayer(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=2,
            act="relu")

        self.layer1 = nn.Sequential(*[
            BottleneckBlock(
                num_channels=64 if i == 0 else 256,
                num_filters=64,
                has_se=has_se,
                stride=1,
                downsample=True if i == 0 else False) for i in range(4)
        ])

        self.conv_tr1_1 = ConvBNLayer(
            num_channels=256, num_filters=width, filter_size=3)
        self.conv_tr1_2 = ConvBNLayer(
            num_channels=256, num_filters=width * 2, filter_size=3, stride=2)

        self.st2 = Stage(
            num_modules=1, num_filters=channels_2, has_se=self.has_se)

        self.conv_tr2 = ConvBNLayer(
            num_channels=width * 2,
            num_filters=width * 4,
            filter_size=3,
            stride=2)
        self.st3 = Stage(
            num_modules=4, num_filters=channels_3, has_se=self.has_se)

        self.conv_tr3 = ConvBNLayer(
            num_channels=width * 4,
            num_filters=width * 8,
            filter_size=3,
            stride=2)

        self.st4 = Stage(
            num_modules=3, num_filters=channels_4, has_se=self.has_se)

        # classification
        num_filters_list = [32, 64, 128, 256]
        self.last_cls = LastClsOut(
            num_channel_list=channels_4,
            has_se=self.has_se,
            num_filters_list=num_filters_list)

        last_num_filters = [256, 512, 1024]
        self.cls_head_conv_list = nn.LayerList()
        for idx in range(3):
            self.cls_head_conv_list.append(
                ConvBNLayer(
                    num_channels=num_filters_list[idx] * 4,
                    num_filters=last_num_filters[idx],
                    filter_size=3,
                    stride=2))

        self.conv_last = ConvBNLayer(
            num_channels=1024, num_filters=2048, filter_size=1, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2D(1)

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.fc = nn.Linear(
            2048,
            class_num,
            weight_attr=ParamAttr(initializer=Uniform(-stdv, stdv)))
        if return_patterns is not None:
            self.update_res(return_patterns)
            self.register_forward_post_hook(self._return_dict_hook)

    def forward(self, x):
        x = self.conv_layer1_1(x)
        x = self.conv_layer1_2(x)

        x = self.layer1(x)

        tr1_1 = self.conv_tr1_1(x)
        tr1_2 = self.conv_tr1_2(x)
        x = self.st2([tr1_1, tr1_2])

        tr2 = self.conv_tr2(x[-1])
        x.append(tr2)
        x = self.st3(x)

        tr3 = self.conv_tr3(x[-1])
        x.append(tr3)
        x = self.st4(x)

        x = self.last_cls(x)

        y = x[0]
        for idx in range(3):
            y = paddle.add(x[idx + 1], self.cls_head_conv_list[idx](y))

        y = self.conv_last(y)
        y = self.avg_pool(y)
        y = paddle.reshape(y, shape=[-1, y.shape[1]])
        y = self.fc(y)
        return y


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def HRNet_W18_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W18_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W18_C` model depends on args.
    """
    model = HRNet(width=18, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W18_C"], use_ssld)
    return model


def HRNet_W30_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W30_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W30_C` model depends on args.
    """
    model = HRNet(width=30, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W30_C"], use_ssld)
    return model


def HRNet_W32_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W32_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W32_C` model depends on args.
    """
    model = HRNet(width=32, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W32_C"], use_ssld)
    return model


def HRNet_W40_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W40_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W40_C` model depends on args.
    """
    model = HRNet(width=40, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W40_C"], use_ssld)
    return model


def HRNet_W44_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W44_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W44_C` model depends on args.
    """
    model = HRNet(width=44, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W44_C"], use_ssld)
    return model


def HRNet_W48_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W48_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W48_C` model depends on args.
    """
    model = HRNet(width=48, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W48_C"], use_ssld)
    return model


def HRNet_W60_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W60_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W60_C` model depends on args.
    """
    model = HRNet(width=60, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W60_C"], use_ssld)
    return model


def HRNet_W64_C(pretrained=False, use_ssld=False, **kwargs):
    """
    HRNet_W64_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `HRNet_W64_C` model depends on args.
    """
    model = HRNet(width=64, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["HRNet_W64_C"], use_ssld)
    return model


def SE_HRNet_W18_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W18_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W18_C` model depends on args.
    """
    model = HRNet(width=18, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W18_C"], use_ssld)
    return model


def SE_HRNet_W30_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W30_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W30_C` model depends on args.
    """
    model = HRNet(width=30, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W30_C"], use_ssld)
    return model


def SE_HRNet_W32_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W32_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W32_C` model depends on args.
    """
    model = HRNet(width=32, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W32_C"], use_ssld)
    return model


def SE_HRNet_W40_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W40_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W40_C` model depends on args.
    """
    model = HRNet(width=40, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W40_C"], use_ssld)
    return model


def SE_HRNet_W44_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W44_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W44_C` model depends on args.
    """
    model = HRNet(width=44, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W44_C"], use_ssld)
    return model


def SE_HRNet_W48_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W48_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W48_C` model depends on args.
    """
    model = HRNet(width=48, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W48_C"], use_ssld)
    return model


def SE_HRNet_W60_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W60_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W60_C` model depends on args.
    """
    model = HRNet(width=60, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W60_C"], use_ssld)
    return model


def SE_HRNet_W64_C(pretrained=False, use_ssld=False, **kwargs):
    """
    SE_HRNet_W64_C
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `SE_HRNet_W64_C` model depends on args.
    """
    model = HRNet(width=64, has_se=True, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["SE_HRNet_W64_C"], use_ssld)
    return model
