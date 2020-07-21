# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import six
import math

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay


class DarkNet(object):
    """
    DarkNet, see https://pjreddie.com/darknet/yolo/
    Args:
        depth (int): network depth, currently only darknet 53 is supported
        norm_type (str): normalization type, 'bn' and 'sync_bn' are supported
        norm_decay (float): weight decay for normalization layer weights
    """

    def __init__(self,
                 depth=53,
                 num_classes=None,
                 norm_type='bn',
                 norm_decay=0.,
                 bn_act='leaky',
                 weight_prefix_name=''):
        assert depth in [53], "unsupported depth value"
        self.depth = depth
        self.num_classes = num_classes
        self.norm_type = norm_type
        self.norm_decay = norm_decay
        self.depth_cfg = {53: ([1, 2, 8, 8, 4], self.basicblock)}
        self.bn_act = bn_act
        self.prefix_name = weight_prefix_name

    def _conv_norm(self,
                   input,
                   ch_out,
                   filter_size,
                   stride,
                   padding,
                   act='leaky',
                   name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        if self.num_classes:
            regularizer = None
        else:
            regularizer = L2Decay(float(self.norm_decay))
        bn_param_attr = ParamAttr(
            regularizer=regularizer, name=bn_name + '.scale')
        bn_bias_attr = ParamAttr(
            regularizer=regularizer, name=bn_name + '.offset')

        out = fluid.layers.batch_norm(
            input=conv,
            param_attr=bn_param_attr,
            bias_attr=bn_bias_attr,
            moving_mean_name=bn_name + '.mean',
            moving_variance_name=bn_name + '.var')

        # leaky relu here has `alpha` as 0.1, can not be set by
        # `act` param in fluid.layers.batch_norm above.
        if act == 'leaky':
            out = fluid.layers.leaky_relu(x=out, alpha=0.1)
        if act == 'relu':
            out = fluid.layers.relu(x=out)
        return out

    def _downsample(self,
                    input,
                    ch_out,
                    filter_size=3,
                    stride=2,
                    padding=1,
                    name=None):
        return self._conv_norm(
            input,
            ch_out=ch_out,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=self.bn_act,
            name=name)

    def basicblock(self, input, ch_out, name=None):
        conv1 = self._conv_norm(
            input,
            ch_out=ch_out,
            filter_size=1,
            stride=1,
            padding=0,
            act=self.bn_act,
            name=name + ".0")
        conv2 = self._conv_norm(
            conv1,
            ch_out=ch_out * 2,
            filter_size=3,
            stride=1,
            padding=1,
            act=self.bn_act,
            name=name + ".1")
        out = fluid.layers.elementwise_add(x=input, y=conv2, act=None)
        return out

    def layer_warp(self, block_func, input, ch_out, count, name=None):
        out = block_func(input, ch_out=ch_out, name='{}.0'.format(name))
        for j in six.moves.xrange(1, count):
            out = block_func(out, ch_out=ch_out, name='{}.{}'.format(name, j))
        return out

    def __call__(self, input):
        """
        Get the backbone of DarkNet, that is output for the 5 stages.
        Args:
            input (Variable): input variable.
        Returns:
            The last variables of each stage.
        """
        stages, block_func = self.depth_cfg[self.depth]
        stages = stages[0:5]
        conv = self._conv_norm(
            input=input,
            ch_out=32,
            filter_size=3,
            stride=1,
            padding=1,
            act=self.bn_act,
            name=self.prefix_name + "yolo_input")
        downsample_ = self._downsample(
            input=conv,
            ch_out=conv.shape[1] * 2,
            name=self.prefix_name + "yolo_input.downsample")
        blocks = []
        for i, stage in enumerate(stages):
            block = self.layer_warp(
                block_func=block_func,
                input=downsample_,
                ch_out=32 * 2**i,
                count=stage,
                name=self.prefix_name + "stage.{}".format(i))
            blocks.append(block)
            if i < len(stages) - 1:  # do not downsaple in the last stage
                downsample_ = self._downsample(
                    input=block,
                    ch_out=block.shape[1] * 2,
                    name=self.prefix_name + "stage.{}.downsample".format(i))
        if self.num_classes is not None:
            pool = fluid.layers.pool2d(
                input=blocks[-1], pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=self.num_classes,
                param_attr=ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv),
                    name='fc_weights'),
                bias_attr=ParamAttr(name='fc_offset'))
            return out

        return blocks
