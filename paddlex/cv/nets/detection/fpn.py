# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import copy
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.regularizer import L2Decay

__all__ = ['FPN', 'HRFPN']


def ConvNorm(input,
             num_filters,
             filter_size,
             stride=1,
             groups=1,
             norm_decay=0.,
             norm_type='affine_channel',
             norm_groups=32,
             dilation=1,
             lr_scale=1,
             freeze_norm=False,
             act=None,
             norm_name=None,
             initializer=None,
             name=None):
    fan = num_filters
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=((filter_size - 1) // 2) * dilation,
        dilation=dilation,
        groups=groups,
        act=None,
        param_attr=ParamAttr(
            name=name + "_weights",
            initializer=initializer,
            learning_rate=lr_scale),
        bias_attr=False,
        name=name + '.conv2d.output.1')

    norm_lr = 0. if freeze_norm else 1.
    pattr = ParamAttr(
        name=norm_name + '_scale',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))
    battr = ParamAttr(
        name=norm_name + '_offset',
        learning_rate=norm_lr * lr_scale,
        regularizer=L2Decay(norm_decay))

    if norm_type in ['bn', 'sync_bn']:
        global_stats = True if freeze_norm else False
        out = fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            param_attr=pattr,
            bias_attr=battr,
            moving_mean_name=norm_name + '_mean',
            moving_variance_name=norm_name + '_variance',
            use_global_stats=global_stats)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'gn':
        out = fluid.layers.group_norm(
            input=conv,
            act=act,
            name=norm_name + '.output.1',
            groups=norm_groups,
            param_attr=pattr,
            bias_attr=battr)
        scale = fluid.framework._get_var(pattr.name)
        bias = fluid.framework._get_var(battr.name)
    elif norm_type == 'affine_channel':
        scale = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=pattr,
            default_initializer=fluid.initializer.Constant(1.))
        bias = fluid.layers.create_parameter(
            shape=[conv.shape[1]],
            dtype=conv.dtype,
            attr=battr,
            default_initializer=fluid.initializer.Constant(0.))
        out = fluid.layers.affine_channel(
            x=conv, scale=scale, bias=bias, act=act)
    if freeze_norm:
        scale.stop_gradient = True
        bias.stop_gradient = True
    return out


class FPN(object):
    """
    Feature Pyramid Network, see https://arxiv.org/abs/1612.03144

    Args:
        num_chan (int): number of feature channels
        min_level (int): lowest level of the backbone feature map to use
        max_level (int): highest level of the backbone feature map to use
        spatial_scale (list): feature map scaling factor
        has_extra_convs (bool): whether has extral convolutions in higher levels
        norm_type (str|None): normalization type, 'bn'/'sync_bn'/'affine_channel'
    """

    def __init__(self,
                 num_chan=256,
                 min_level=2,
                 max_level=6,
                 spatial_scale=[1. / 32., 1. / 16., 1. / 8., 1. / 4.],
                 has_extra_convs=False,
                 norm_type=None,
                 freeze_norm=False):
        self.freeze_norm = freeze_norm
        self.num_chan = num_chan
        self.min_level = min_level
        self.max_level = max_level
        self.spatial_scale = spatial_scale
        self.has_extra_convs = has_extra_convs
        self.norm_type = norm_type

    def _add_topdown_lateral(self, body_name, body_input, upper_output):
        lateral_name = 'fpn_inner_' + body_name + '_lateral'
        topdown_name = 'fpn_topdown_' + body_name
        fan = body_input.shape[1]
        if self.norm_type:
            initializer = Xavier(fan_out=fan)
            lateral = ConvNorm(
                body_input,
                self.num_chan,
                1,
                initializer=initializer,
                norm_type=self.norm_type,
                freeze_norm=self.freeze_norm,
                name=lateral_name,
                norm_name=lateral_name)
        else:
            lateral = fluid.layers.conv2d(
                body_input,
                self.num_chan,
                1,
                param_attr=ParamAttr(
                    name=lateral_name + "_w", initializer=Xavier(fan_out=fan)),
                bias_attr=ParamAttr(
                    name=lateral_name + "_b",
                    learning_rate=2.,
                    regularizer=L2Decay(0.)),
                name=lateral_name)
        topdown = fluid.layers.resize_nearest(
            upper_output, scale=2., name=topdown_name)

        return lateral + topdown

    def get_output(self, body_dict):
        """
        Add FPN onto backbone.

        Args:
            body_dict(OrderedDict): Dictionary of variables and each element is the
                output of backbone.

        Return:
            fpn_dict(OrderedDict): A dictionary represents the output of FPN with
                their name.
            spatial_scale(list): A list of multiplicative spatial scale factor.
        """
        spatial_scale = copy.deepcopy(self.spatial_scale)
        body_name_list = list(body_dict.keys())[::-1]
        num_backbone_stages = len(body_name_list)
        self.fpn_inner_output = [[] for _ in range(num_backbone_stages)]
        fpn_inner_name = 'fpn_inner_' + body_name_list[0]
        body_input = body_dict[body_name_list[0]]
        fan = body_input.shape[1]
        if self.norm_type:
            initializer = Xavier(fan_out=fan)
            self.fpn_inner_output[0] = ConvNorm(
                body_input,
                self.num_chan,
                1,
                initializer=initializer,
                norm_type=self.norm_type,
                freeze_norm=self.freeze_norm,
                name=fpn_inner_name,
                norm_name=fpn_inner_name)
        else:
            self.fpn_inner_output[0] = fluid.layers.conv2d(
                body_input,
                self.num_chan,
                1,
                param_attr=ParamAttr(
                    name=fpn_inner_name + "_w",
                    initializer=Xavier(fan_out=fan)),
                bias_attr=ParamAttr(
                    name=fpn_inner_name + "_b",
                    learning_rate=2.,
                    regularizer=L2Decay(0.)),
                name=fpn_inner_name)
        for i in range(1, num_backbone_stages):
            body_name = body_name_list[i]
            body_input = body_dict[body_name]
            top_output = self.fpn_inner_output[i - 1]
            fpn_inner_single = self._add_topdown_lateral(body_name, body_input,
                                                         top_output)
            self.fpn_inner_output[i] = fpn_inner_single
        fpn_dict = {}
        fpn_name_list = []
        for i in range(num_backbone_stages):
            fpn_name = 'fpn_' + body_name_list[i]
            fan = self.fpn_inner_output[i].shape[1] * 3 * 3
            if self.norm_type:
                initializer = Xavier(fan_out=fan)
                fpn_output = ConvNorm(
                    self.fpn_inner_output[i],
                    self.num_chan,
                    3,
                    initializer=initializer,
                    norm_type=self.norm_type,
                    freeze_norm=self.freeze_norm,
                    name=fpn_name,
                    norm_name=fpn_name)
            else:
                fpn_output = fluid.layers.conv2d(
                    self.fpn_inner_output[i],
                    self.num_chan,
                    filter_size=3,
                    padding=1,
                    param_attr=ParamAttr(
                        name=fpn_name + "_w", initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        name=fpn_name + "_b",
                        learning_rate=2.,
                        regularizer=L2Decay(0.)),
                    name=fpn_name)
            fpn_dict[fpn_name] = fpn_output
            fpn_name_list.append(fpn_name)
        if not self.has_extra_convs and self.max_level - self.min_level == len(
                spatial_scale):
            body_top_name = fpn_name_list[0]
            body_top_extension = fluid.layers.pool2d(
                fpn_dict[body_top_name],
                1,
                'max',
                pool_stride=2,
                name=body_top_name + '_subsampled_2x')
            fpn_dict[body_top_name + '_subsampled_2x'] = body_top_extension
            fpn_name_list.insert(0, body_top_name + '_subsampled_2x')
            spatial_scale.insert(0, spatial_scale[0] * 0.5)
        # Coarser FPN levels introduced for RetinaNet
        highest_backbone_level = self.min_level + len(spatial_scale) - 1
        if self.has_extra_convs and self.max_level > highest_backbone_level:
            fpn_blob = body_dict[body_name_list[0]]
            for i in range(highest_backbone_level + 1, self.max_level + 1):
                fpn_blob_in = fpn_blob
                fpn_name = 'fpn_' + str(i)
                if i > highest_backbone_level + 1:
                    fpn_blob_in = fluid.layers.relu(fpn_blob)
                fan = fpn_blob_in.shape[1] * 3 * 3
                fpn_blob = fluid.layers.conv2d(
                    input=fpn_blob_in,
                    num_filters=self.num_chan,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    param_attr=ParamAttr(
                        name=fpn_name + "_w", initializer=Xavier(fan_out=fan)),
                    bias_attr=ParamAttr(
                        name=fpn_name + "_b",
                        learning_rate=2.,
                        regularizer=L2Decay(0.)),
                    name=fpn_name)
                fpn_dict[fpn_name] = fpn_blob
                fpn_name_list.insert(0, fpn_name)
                spatial_scale.insert(0, spatial_scale[0] * 0.5)
        res_dict = OrderedDict([(k, fpn_dict[k]) for k in fpn_name_list])
        return res_dict, spatial_scale


class HRFPN(object):
    """
    HRNet, see https://arxiv.org/abs/1908.07919

    Args:
        num_chan (int): number of feature channels
        pooling_type (str): pooling type of downsampling
        share_conv (bool): whethet to share conv for different layers' reduction
        spatial_scale (list): feature map scaling factor
    """

    def __init__(
            self,
            num_chan=256,
            pooling_type="avg",
            share_conv=False,
            spatial_scale=[1. / 64, 1. / 32, 1. / 16, 1. / 8, 1. / 4], ):
        self.num_chan = num_chan
        self.pooling_type = pooling_type
        self.share_conv = share_conv
        self.spatial_scale = spatial_scale

    def get_output(self, body_dict):
        num_out = len(self.spatial_scale)
        body_name_list = list(body_dict.keys())

        num_backbone_stages = len(body_name_list)

        outs = []
        outs.append(body_dict[body_name_list[0]])

        # resize
        for i in range(1, len(body_dict)):
            resized = self.resize_input_tensor(body_dict[body_name_list[i]],
                                               outs[0], 2**i)
            outs.append(resized)

        # concat
        out = fluid.layers.concat(outs, axis=1)

        # reduction
        out = fluid.layers.conv2d(
            input=out,
            num_filters=self.num_chan,
            filter_size=1,
            stride=1,
            padding=0,
            param_attr=ParamAttr(name='hrfpn_reduction_weights'),
            bias_attr=False)

        # conv
        outs = [out]
        for i in range(1, num_out):
            outs.append(
                self.pooling(
                    out,
                    size=2**i,
                    stride=2**i,
                    pooling_type=self.pooling_type))
        outputs = []

        for i in range(num_out):
            conv_name = "shared_fpn_conv" if self.share_conv else "shared_fpn_conv_" + str(
                i)
            conv = fluid.layers.conv2d(
                input=outs[i],
                num_filters=self.num_chan,
                filter_size=3,
                stride=1,
                padding=1,
                param_attr=ParamAttr(name=conv_name + "_weights"),
                bias_attr=False)
            outputs.append(conv)

        for idx in range(0, num_out - len(body_name_list)):
            body_name_list.append("fpn_res5_sum_subsampled_{}x".format(2**(
                idx + 1)))

        outputs = outputs[::-1]
        body_name_list = body_name_list[::-1]

        res_dict = OrderedDict([(body_name_list[k], outputs[k])
                                for k in range(len(body_name_list))])
        return res_dict, self.spatial_scale

    def resize_input_tensor(self, body_input, ref_output, scale):
        shape = fluid.layers.shape(ref_output)
        shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
        out_shape_ = shape_hw
        out_shape = fluid.layers.cast(out_shape_, dtype='int32')
        out_shape.stop_gradient = True
        body_output = fluid.layers.resize_bilinear(
            body_input, scale=scale, out_shape=out_shape)
        return body_output

    def pooling(self, input, size, stride, pooling_type):
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=size,
            pool_stride=stride,
            pool_type=pooling_type)
        return pool
