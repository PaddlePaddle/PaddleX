# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import, division, print_function

import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import MaxPool2D

from paddlex.ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "VGG11":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG11_pretrained.pdparams",
    "VGG13":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG13_pretrained.pdparams",
    "VGG16":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG16_pretrained.pdparams",
    "VGG19":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/VGG19_pretrained.pdparams",
}
__all__ = MODEL_URLS.keys()

# VGG config
# key: VGG network depth
# value: conv num in different blocks
NET_CONFIG = {
    11: [1, 1, 2, 2, 2],
    13: [2, 2, 2, 2, 2],
    16: [2, 2, 3, 3, 3],
    19: [2, 2, 4, 4, 4]
}


class ConvBlock(TheseusLayer):
    def __init__(self, input_channels, output_channels, groups):
        super().__init__()

        self.groups = groups
        self.conv1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self.conv2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)
        if groups == 3 or groups == 4:
            self.conv3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)
        if groups == 4:
            self.conv4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)

        self.max_pool = MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self.conv2(x)
            x = self.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self.conv3(x)
            x = self.relu(x)
        if self.groups == 4:
            x = self.conv4(x)
            x = self.relu(x)
        x = self.max_pool(x)
        return x


class VGGNet(TheseusLayer):
    """
    VGGNet
    Args:
        config: list. VGGNet config.
        stop_grad_layers: int=0. The parameters in blocks which index larger than `stop_grad_layers`, will be set `param.trainable=False`
        class_num: int=1000. The number of classes.
    Returns:
        model: nn.Layer. Specific VGG model depends on args.
    """

    def __init__(self,
                 config,
                 stop_grad_layers=0,
                 class_num=1000,
                 return_patterns=None):
        super().__init__()

        self.stop_grad_layers = stop_grad_layers

        self.conv_block_1 = ConvBlock(3, 64, config[0])
        self.conv_block_2 = ConvBlock(64, 128, config[1])
        self.conv_block_3 = ConvBlock(128, 256, config[2])
        self.conv_block_4 = ConvBlock(256, 512, config[3])
        self.conv_block_5 = ConvBlock(512, 512, config[4])

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)

        for idx, block in enumerate([
                self.conv_block_1, self.conv_block_2, self.conv_block_3,
                self.conv_block_4, self.conv_block_5
        ]):
            if self.stop_grad_layers >= idx + 1:
                for param in block.parameters():
                    param.trainable = False

        self.drop = Dropout(p=0.5, mode="downscale_in_infer")
        self.fc1 = Linear(7 * 7 * 512, 4096)
        self.fc2 = Linear(4096, 4096)
        self.fc3 = Linear(4096, class_num)
        if return_patterns is not None:
            self.update_res(return_patterns)
            self.register_forward_post_hook(self._return_dict_hook)

    def forward(self, inputs):
        x = self.conv_block_1(inputs)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


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


def VGG11(pretrained=False, use_ssld=False, **kwargs):
    """
    VGG11
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `VGG11` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[11], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG11"], use_ssld)
    return model


def VGG13(pretrained=False, use_ssld=False, **kwargs):
    """
    VGG13
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `VGG13` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[13], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG13"], use_ssld)
    return model


def VGG16(pretrained=False, use_ssld=False, **kwargs):
    """
    VGG16
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `VGG16` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[16], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG16"], use_ssld)
    return model


def VGG19(pretrained=False, use_ssld=False, **kwargs):
    """
    VGG19
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `VGG19` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[19], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["VGG19"], use_ssld)
    return model
