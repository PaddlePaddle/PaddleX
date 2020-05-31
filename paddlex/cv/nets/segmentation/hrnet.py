# coding: utf8
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

from collections import OrderedDict

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr
from .model_utils.libs import sigmoid_to_softmax
from .model_utils.loss import softmax_with_loss
from .model_utils.loss import dice_loss
from .model_utils.loss import bce_loss
import paddlex
import paddlex.utils.logging as logging


class HRNet(object):
    def __init__(self,
                 num_classes,
                 mode='train',
                 width=18,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255):
        # dice_loss或bce_loss只适用两类分割中
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classfication"
            )

        if class_weight is not None:
            if isinstance(class_weight, list):
                if len(class_weight) != num_classes:
                    raise ValueError(
                        "Length of class_weight should be equal to number of classes"
                    )
            elif isinstance(class_weight, str):
                if class_weight.lower() != 'dynamic':
                    raise ValueError(
                        "if class_weight is string, must be dynamic!")
            else:
                raise TypeError(
                    'Expect class_weight is a list or string but receive {}'.
                    format(type(class_weight)))

        self.num_classes = num_classes
        self.mode = mode
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.backbone = paddlex.cv.nets.hrnet.HRNet(
            width=width, feature_maps="stage4")

    def build_net(self, inputs):
        if self.use_dice_loss or self.use_bce_loss:
            self.num_classes = 1
        image = inputs['image']
        st4 = self.backbone(image)
        # upsample
        shape = fluid.layers.shape(st4[0])[-2:]
        st4[1] = fluid.layers.resize_bilinear(st4[1], out_shape=shape)
        st4[2] = fluid.layers.resize_bilinear(st4[2], out_shape=shape)
        st4[3] = fluid.layers.resize_bilinear(st4[3], out_shape=shape)

        out = fluid.layers.concat(st4, axis=1)
        last_channels = sum(self.backbone.channels[self.backbone.width][-1])

        out = self._conv_bn_layer(
            input=out,
            filter_size=1,
            num_filters=last_channels,
            stride=1,
            if_act=True,
            name='conv-2')
        out = fluid.layers.conv2d(
            input=out,
            num_filters=self.num_classes,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(
                initializer=MSRA(), name='conv-1_weights'),
            bias_attr=False)

        input_shape = fluid.layers.shape(image)[-2:]
        logit = fluid.layers.resize_bilinear(out, input_shape)

        if self.num_classes == 1:
            out = sigmoid_to_softmax(logit)
            out = fluid.layers.transpose(out, [0, 2, 3, 1])
        else:
            out = fluid.layers.transpose(logit, [0, 2, 3, 1])

        pred = fluid.layers.argmax(out, axis=3)
        pred = fluid.layers.unsqueeze(pred, axes=[3])

        if self.mode == 'train':
            label = inputs['label']
            mask = label != self.ignore_index
            return self._get_loss(logit, label, mask)
        elif self.mode == 'eval':
            label = inputs['label']
            mask = label != self.ignore_index
            loss = self._get_loss(logit, label, mask)
            return loss, pred, label, mask
        else:
            if self.num_classes == 1:
                logit = sigmoid_to_softmax(logit)
            else:
                logit = fluid.layers.softmax(logit, axis=1)
            return pred, logit

    def generate_inputs(self):
        inputs = OrderedDict()
        inputs['image'] = fluid.data(
            dtype='float32', shape=[None, 3, None, None], name='image')
        if self.mode == 'train':
            inputs['label'] = fluid.data(
                dtype='int32', shape=[None, 1, None, None], name='label')
        elif self.mode == 'eval':
            inputs['label'] = fluid.data(
                dtype='int32', shape=[None, 1, None, None], name='label')
        return inputs

    def _get_loss(self, logit, label, mask):
        avg_loss = 0
        if not (self.use_dice_loss or self.use_bce_loss):
            avg_loss += softmax_with_loss(
                logit,
                label,
                mask,
                num_classes=self.num_classes,
                weight=self.class_weight,
                ignore_index=self.ignore_index)
        else:
            if self.use_dice_loss:
                avg_loss += dice_loss(logit, label, mask)
            if self.use_bce_loss:
                avg_loss += bce_loss(
                    logit, label, mask, ignore_index=self.ignore_index)

        return avg_loss

    def _conv_bn_layer(self,
                       input,
                       filter_size,
                       num_filters,
                       stride=1,
                       padding=1,
                       num_groups=1,
                       if_act=True,
                       name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=num_groups,
            act=None,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(
                name=bn_name + "_scale",
                initializer=fluid.initializer.Constant(1.0)),
            bias_attr=ParamAttr(
                name=bn_name + "_offset",
                initializer=fluid.initializer.Constant(0.0)),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            bn = fluid.layers.relu(bn)
        return bn
