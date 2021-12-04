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

# Code was heavily based on https://github.com/rwightman/pytorch-image-models

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr

from paddlex.ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "CSPDarkNet53":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSPDarkNet53_pretrained.pdparams"
}

MODEL_CFGS = {
    "CSPDarkNet53": dict(
        stem=dict(
            out_chs=32, kernel_size=3, stride=1, pool=''),
        stage=dict(
            out_chs=(64, 128, 256, 512, 1024),
            depth=(1, 2, 8, 8, 4),
            stride=(2, ) * 5,
            exp_ratio=(2., ) + (1., ) * 4,
            bottle_ratio=(0.5, ) + (1.0, ) * 4,
            block_ratio=(1., ) + (0.5, ) * 4,
            down_growth=True, ))
}

__all__ = ['CSPDarkNet53'
           ]  # model_registry will add each entrypoint fn to this


class ConvBnAct(nn.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 act_layer=nn.LeakyReLU,
                 norm_layer=nn.BatchNorm2D):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=ParamAttr(),
            bias_attr=False)

        self.bn = norm_layer(num_features=output_channels)
        self.act = act_layer()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


def create_stem(in_chans=3,
                out_chs=32,
                kernel_size=3,
                stride=2,
                pool='',
                act_layer=None,
                norm_layer=None):
    stem = nn.Sequential()
    if not isinstance(out_chs, (tuple, list)):
        out_chs = [out_chs]
    assert len(out_chs)
    in_c = in_chans
    for i, out_c in enumerate(out_chs):
        conv_name = f'conv{i + 1}'
        stem.add_sublayer(
            conv_name,
            ConvBnAct(
                in_c,
                out_c,
                kernel_size,
                stride=stride if i == 0 else 1,
                act_layer=act_layer,
                norm_layer=norm_layer))
        in_c = out_c
        last_conv = conv_name
    if pool:
        stem.add_sublayer(
            'pool', nn.MaxPool2D(
                kernel_size=3, stride=2, padding=1))
    return stem, dict(
        num_chs=in_c, reduction=stride, module='.'.join(['stem', last_conv]))


class DarkBlock(nn.Layer):
    def __init__(self,
                 in_chs,
                 out_chs,
                 dilation=1,
                 bottle_ratio=0.5,
                 groups=1,
                 act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2D,
                 attn_layer=None,
                 drop_block=None):
        super(DarkBlock, self).__init__()
        mid_chs = int(round(out_chs * bottle_ratio))
        ckwargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvBnAct(in_chs, mid_chs, kernel_size=1, **ckwargs)
        self.conv2 = ConvBnAct(
            mid_chs,
            out_chs,
            kernel_size=3,
            dilation=dilation,
            groups=groups,
            **ckwargs)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + shortcut
        return x


class CrossStage(nn.Layer):
    def __init__(self,
                 in_chs,
                 out_chs,
                 stride,
                 dilation,
                 depth,
                 block_ratio=1.,
                 bottle_ratio=1.,
                 exp_ratio=1.,
                 groups=1,
                 first_dilation=None,
                 down_growth=False,
                 cross_linear=False,
                 block_dpr=None,
                 block_fn=DarkBlock,
                 **block_kwargs):
        super(CrossStage, self).__init__()
        first_dilation = first_dilation or dilation
        down_chs = out_chs if down_growth else in_chs
        exp_chs = int(round(out_chs * exp_ratio))
        block_out_chs = int(round(out_chs * block_ratio))
        conv_kwargs = dict(
            act_layer=block_kwargs.get('act_layer'),
            norm_layer=block_kwargs.get('norm_layer'))

        if stride != 1 or first_dilation != dilation:
            self.conv_down = ConvBnAct(
                in_chs,
                down_chs,
                kernel_size=3,
                stride=stride,
                dilation=first_dilation,
                groups=groups,
                **conv_kwargs)
            prev_chs = down_chs
        else:
            self.conv_down = None
            prev_chs = in_chs

        self.conv_exp = ConvBnAct(
            prev_chs, exp_chs, kernel_size=1, **conv_kwargs)
        prev_chs = exp_chs // 2  # output of conv_exp is always split in two

        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_sublayer(
                str(i),
                block_fn(prev_chs, block_out_chs, dilation, bottle_ratio,
                         groups, **block_kwargs))
            prev_chs = block_out_chs

        # transition convs
        self.conv_transition_b = ConvBnAct(
            prev_chs, exp_chs // 2, kernel_size=1, **conv_kwargs)
        self.conv_transition = ConvBnAct(
            exp_chs, out_chs, kernel_size=1, **conv_kwargs)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        x = self.conv_exp(x)
        split = x.shape[1] // 2
        xs, xb = x[:, :split], x[:, split:]
        xb = self.blocks(xb)
        xb = self.conv_transition_b(xb)
        out = self.conv_transition(paddle.concat([xs, xb], axis=1))
        return out


class DarkStage(nn.Layer):
    def __init__(self,
                 in_chs,
                 out_chs,
                 stride,
                 dilation,
                 depth,
                 block_ratio=1.,
                 bottle_ratio=1.,
                 groups=1,
                 first_dilation=None,
                 block_fn=DarkBlock,
                 block_dpr=None,
                 **block_kwargs):
        super().__init__()
        first_dilation = first_dilation or dilation

        self.conv_down = ConvBnAct(
            in_chs,
            out_chs,
            kernel_size=3,
            stride=stride,
            dilation=first_dilation,
            groups=groups,
            act_layer=block_kwargs.get('act_layer'),
            norm_layer=block_kwargs.get('norm_layer'))

        prev_chs = out_chs
        block_out_chs = int(round(out_chs * block_ratio))
        self.blocks = nn.Sequential()
        for i in range(depth):
            self.blocks.add_sublayer(
                str(i),
                block_fn(prev_chs, block_out_chs, dilation, bottle_ratio,
                         groups, **block_kwargs))
            prev_chs = block_out_chs

    def forward(self, x):
        x = self.conv_down(x)
        x = self.blocks(x)
        return x


def _cfg_to_stage_args(cfg, curr_stride=2, output_stride=32):
    # get per stage args for stage and containing blocks, calculate strides to meet target output_stride
    num_stages = len(cfg['depth'])
    if 'groups' not in cfg:
        cfg['groups'] = (1, ) * num_stages
    if 'down_growth' in cfg and not isinstance(cfg['down_growth'],
                                               (list, tuple)):
        cfg['down_growth'] = (cfg['down_growth'], ) * num_stages
    stage_strides = []
    stage_dilations = []
    stage_first_dilations = []
    dilation = 1
    for cfg_stride in cfg['stride']:
        stage_first_dilations.append(dilation)
        if curr_stride >= output_stride:
            dilation *= cfg_stride
            stride = 1
        else:
            stride = cfg_stride
            curr_stride *= stride
        stage_strides.append(stride)
        stage_dilations.append(dilation)
    cfg['stride'] = stage_strides
    cfg['dilation'] = stage_dilations
    cfg['first_dilation'] = stage_first_dilations
    stage_args = [
        dict(zip(cfg.keys(), values)) for values in zip(*cfg.values())
    ]
    return stage_args


class CSPNet(nn.Layer):
    def __init__(self,
                 cfg,
                 in_chans=3,
                 class_num=1000,
                 output_stride=32,
                 global_pool='avg',
                 drop_rate=0.,
                 act_layer=nn.LeakyReLU,
                 norm_layer=nn.BatchNorm2D,
                 zero_init_last_bn=True,
                 stage_fn=CrossStage,
                 block_fn=DarkBlock):
        super().__init__()
        self.class_num = class_num
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        layer_args = dict(act_layer=act_layer, norm_layer=norm_layer)

        # Construct the stem
        self.stem, stem_feat_info = create_stem(in_chans, **cfg['stem'],
                                                **layer_args)
        self.feature_info = [stem_feat_info]
        prev_chs = stem_feat_info['num_chs']
        curr_stride = stem_feat_info[
            'reduction']  # reduction does not include pool
        if cfg['stem']['pool']:
            curr_stride *= 2

        # Construct the stages
        per_stage_args = _cfg_to_stage_args(
            cfg['stage'], curr_stride=curr_stride, output_stride=output_stride)
        self.stages = nn.LayerList()
        for i, sa in enumerate(per_stage_args):
            self.stages.add_sublayer(
                str(i),
                stage_fn(
                    prev_chs, **sa, **layer_args, block_fn=block_fn))
            prev_chs = sa['out_chs']
            curr_stride *= sa['stride']
            self.feature_info += [
                dict(
                    num_chs=prev_chs,
                    reduction=curr_stride,
                    module=f'stages.{i}')
            ]

        # Construct the head
        self.num_features = prev_chs

        self.pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(
            prev_chs,
            class_num,
            weight_attr=ParamAttr(),
            bias_attr=ParamAttr())

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
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


def CSPDarkNet53(pretrained=False, use_ssld=False, **kwargs):
    model = CSPNet(MODEL_CFGS["CSPDarkNet53"], block_fn=DarkBlock, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["CSPDarkNet53"], use_ssld=use_ssld)
    return model
