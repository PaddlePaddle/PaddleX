# coding: utf8
# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from .model_utils.libs import scope
from .model_utils.libs import bn, bn_relu, relu, conv_bn_layer
from .model_utils.libs import conv, avg_pool
from .model_utils.libs import separate_conv
from .model_utils.libs import sigmoid_to_softmax
from .model_utils.loss import softmax_with_loss
from .model_utils.loss import dice_loss
from .model_utils.loss import bce_loss


class FastSCNN(object):
    def __init__(self,
                 num_classes,
                 input_channel=3,
                 mode='train',
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 multi_loss_weight=[1.0],
                 ignore_index=255,
                 fixed_input_shape=None):
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
        self.input_channel = input_channel
        self.mode = mode
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.multi_loss_weight = multi_loss_weight
        self.fixed_input_shape = fixed_input_shape

    def build_net(self, inputs):
        if self.use_dice_loss or self.use_bce_loss:
            self.num_classes = 1
        image = inputs['image']
        size = fluid.layers.shape(image)[2:]
        with scope('learning_to_downsample'):
            higher_res_features = self._learning_to_downsample(image, 32, 48,
                                                               64)
        with scope('global_feature_extractor'):
            lower_res_feature = self._global_feature_extractor(
                higher_res_features, 64, [64, 96, 128], 128, 6, [3, 3, 3])
        with scope('feature_fusion'):
            x = self._feature_fusion(higher_res_features, lower_res_feature,
                                     64, 128, 128)
        with scope('classifier'):
            logit = self._classifier(x, 128)
            logit = fluid.layers.resize_bilinear(logit, size, align_mode=0)

        if len(self.multi_loss_weight) == 3:
            with scope('aux_layer_higher'):
                higher_logit = self._aux_layer(higher_res_features,
                                               self.num_classes)
                higher_logit = fluid.layers.resize_bilinear(
                    higher_logit, size, align_mode=0)
            with scope('aux_layer_lower'):
                lower_logit = self._aux_layer(lower_res_feature,
                                              self.num_classes)
                lower_logit = fluid.layers.resize_bilinear(
                    lower_logit, size, align_mode=0)
            logit = (logit, higher_logit, lower_logit)
        elif len(self.multi_loss_weight) == 2:
            with scope('aux_layer_higher'):
                higher_logit = self._aux_layer(higher_res_features,
                                               self.num_classes)
                higher_logit = fluid.layers.resize_bilinear(
                    higher_logit, size, align_mode=0)
            logit = (logit, higher_logit)
        else:
            logit = (logit, )

        if self.num_classes == 1:
            out = sigmoid_to_softmax(logit[0])
            out = fluid.layers.transpose(out, [0, 2, 3, 1])
        else:
            out = fluid.layers.transpose(logit[0], [0, 2, 3, 1])

        pred = fluid.layers.argmax(out, axis=3)
        pred = fluid.layers.unsqueeze(pred, axes=[3])

        if self.mode == 'train':
            label = inputs['label']
            return self._get_loss(logit, label)
        elif self.mode == 'eval':
            label = inputs['label']
            loss = self._get_loss(logit, label)
            return loss, pred, label, mask
        else:
            if self.num_classes == 1:
                logit = sigmoid_to_softmax(logit[0])
            else:
                logit = fluid.layers.softmax(logit[0], axis=1)
            return pred, logit

    def generate_inputs(self):
        inputs = OrderedDict()
        if self.fixed_input_shape is not None:
            input_shape = [
                None, self.input_channel, self.fixed_input_shape[1],
                self.fixed_input_shape[0]
            ]
            inputs['image'] = fluid.data(
                dtype='float32', shape=input_shape, name='image')
        else:
            inputs['image'] = fluid.data(
                dtype='float32',
                shape=[None, self.input_channel, None, None],
                name='image')
        if self.mode == 'train':
            inputs['label'] = fluid.data(
                dtype='int32', shape=[None, 1, None, None], name='label')
        elif self.mode == 'eval':
            inputs['label'] = fluid.data(
                dtype='int32', shape=[None, 1, None, None], name='label')
        return inputs

    def _get_loss(self, logits, label):
        avg_loss = 0
        if not (self.use_dice_loss or self.use_bce_loss):
            for i, logit in enumerate(logits):
                logit_mask = (
                    label.astype('int32') != self.ignore_index).astype('int32')
                loss = softmax_with_loss(
                    logit,
                    label,
                    logit_mask,
                    num_classes=self.num_classes,
                    weight=self.class_weight,
                    ignore_index=self.ignore_index)
                avg_loss += self.multi_loss_weight[i] * loss
        else:
            if self.use_dice_loss:
                for i, logit in enumerate(logits):
                    logit_mask = (label.astype('int32') != self.ignore_index
                                  ).astype('int32')
                    loss = dice_loss(logit, label, logit_mask)
                    avg_loss += self.multi_loss_weight[i] * loss
            if self.use_bce_loss:
                for i, logit in enumerate(logits):
                    #logit_label = fluid.layers.resize_nearest(label, logit_shape[2:])
                    logit_mask = (label.astype('int32') != self.ignore_index
                                  ).astype('int32')
                    loss = bce_loss(
                        logit,
                        label,
                        logit_mask,
                        ignore_index=self.ignore_index)
                    avg_loss += self.multi_loss_weight[i] * loss
        return avg_loss

    def _learning_to_downsample(self,
                                x,
                                dw_channels1=32,
                                dw_channels2=48,
                                out_channels=64):
        x = relu(bn(conv(x, dw_channels1, 3, 2)))
        with scope('dsconv1'):
            x = separate_conv(
                x, dw_channels2, stride=2, filter=3, act=fluid.layers.relu)
        with scope('dsconv2'):
            x = separate_conv(
                x, out_channels, stride=2, filter=3, act=fluid.layers.relu)
        return x

    def _shortcut(self, input, data_residual):
        return fluid.layers.elementwise_add(input, data_residual)

    def _dropout2d(self, input, prob, is_train=False):
        if not is_train:
            return input
        keep_prob = 1.0 - prob
        shape = fluid.layers.shape(input)
        channels = shape[1]
        random_tensor = keep_prob + fluid.layers.uniform_random(
            [shape[0], channels, 1, 1], min=0., max=1.)
        binary_tensor = fluid.layers.floor(random_tensor)
        output = input / keep_prob * binary_tensor
        return output

    def _inverted_residual_unit(self,
                                input,
                                num_in_filter,
                                num_filters,
                                ifshortcut,
                                stride,
                                filter_size,
                                padding,
                                expansion_factor,
                                name=None):
        num_expfilter = int(round(num_in_filter * expansion_factor))

        channel_expand = conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name=name + '_expand')

        bottleneck_conv = conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            if_act=True,
            name=name + '_dwise',
            use_cudnn=False)

        depthwise_output = bottleneck_conv

        linear_out = conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=False,
            name=name + '_linear')

        if ifshortcut:
            out = self._shortcut(input=input, data_residual=linear_out)
            return out, depthwise_output
        else:
            return linear_out, depthwise_output

    def _inverted_blocks(self, input, in_c, t, c, n, s, name=None):
        first_block, depthwise_output = self._inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + '_1')

        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block, depthwise_output = self._inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=3,
                padding=1,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block, depthwise_output

    def _psp_module(self, input, out_features):

        cat_layers = []
        sizes = (1, 2, 3, 6)
        for size in sizes:
            psp_name = "psp" + str(size)
            with scope(psp_name):
                pool = fluid.layers.adaptive_pool2d(
                    input,
                    pool_size=[size, size],
                    pool_type='avg',
                    name=psp_name + '_adapool')
                data = conv(
                    pool,
                    out_features,
                    filter_size=1,
                    bias_attr=False,
                    name=psp_name + '_conv')
                data_bn = bn(data, act='relu')
                interp = fluid.layers.resize_bilinear(
                    data_bn,
                    out_shape=fluid.layers.shape(input)[2:],
                    name=psp_name + '_interp',
                    align_mode=0)
            cat_layers.append(interp)
        cat_layers = [input] + cat_layers
        out = fluid.layers.concat(cat_layers, axis=1, name='psp_cat')

        return out

    def _aux_layer(self, x, num_classes):
        x = relu(bn(conv(x, 32, 3, padding=1)))
        x = self._dropout2d(x, 0.1, is_train=(self.mode == 'train'))
        with scope('logit'):
            x = conv(x, num_classes, 1, bias_attr=True)
        return x

    def _feature_fusion(self,
                        higher_res_feature,
                        lower_res_feature,
                        higher_in_channels,
                        lower_in_channels,
                        out_channels,
                        scale_factor=4):
        shape = fluid.layers.shape(higher_res_feature)
        w = shape[-1]
        h = shape[-2]
        lower_res_feature = fluid.layers.resize_bilinear(
            lower_res_feature, [h, w], align_mode=0)

        with scope('dwconv'):
            lower_res_feature = relu(
                bn(conv(lower_res_feature, out_channels,
                        1)))  #(lower_res_feature)
        with scope('conv_lower_res'):
            lower_res_feature = bn(
                conv(
                    lower_res_feature, out_channels, 1, bias_attr=True))
        with scope('conv_higher_res'):
            higher_res_feature = bn(
                conv(
                    higher_res_feature, out_channels, 1, bias_attr=True))
        out = higher_res_feature + lower_res_feature

        return relu(out)

    def _global_feature_extractor(self,
                                  x,
                                  in_channels=64,
                                  block_channels=(64, 96, 128),
                                  out_channels=128,
                                  t=6,
                                  num_blocks=(3, 3, 3)):
        x, _ = self._inverted_blocks(x, in_channels, t, block_channels[0],
                                     num_blocks[0], 2, 'inverted_block_1')
        x, _ = self._inverted_blocks(x, block_channels[0], t,
                                     block_channels[1], num_blocks[1], 2,
                                     'inverted_block_2')
        x, _ = self._inverted_blocks(x, block_channels[1], t,
                                     block_channels[2], num_blocks[2], 1,
                                     'inverted_block_3')
        x = self._psp_module(x, block_channels[2] // 4)

        with scope('out'):
            x = relu(bn(conv(x, out_channels, 1)))

        return x

    def _classifier(self, x, dw_channels, stride=1):
        with scope('dsconv1'):
            x = separate_conv(
                x, dw_channels, stride=stride, filter=3, act=fluid.layers.relu)
        with scope('dsconv2'):
            x = separate_conv(
                x, dw_channels, stride=stride, filter=3, act=fluid.layers.relu)

        x = self._dropout2d(x, 0.1, is_train=self.mode == 'train')
        x = conv(x, self.num_classes, 1, bias_attr=True)
        return x
