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
from .model_utils.libs import scope, name_scope
from .model_utils.libs import bn, bn_relu, relu, qsigmoid
from .model_utils.libs import conv, max_pool, deconv
from .model_utils.libs import separate_conv
from .model_utils.libs import sigmoid_to_softmax
from .model_utils.loss import softmax_with_loss
from .model_utils.loss import dice_loss
from .model_utils.loss import bce_loss
from paddlex.cv.nets.xception import Xception
from paddlex.cv.nets.mobilenet_v2 import MobileNetV2


class DeepLabv3p(object):
    """实现DeepLabv3+模型
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1802.02611>`

    Args:
        num_classes (int): 类别数。
        backbone (paddlex.cv.nets): 神经网络，实现DeepLabv3+特征图的计算。
        mode (str): 网络运行模式，根据mode构建网络的输入和返回。
            当mode为'train'时，输入为image(-1, 3, -1, -1)和label (-1, 1, -1, -1) 返回loss。
            当mode为'train'时，输入为image (-1, 3, -1, -1)和label  (-1, 1, -1, -1)，返回loss，
            pred (与网络输入label 相同大小的预测结果，值代表相应的类别），label，mask（非忽略值的mask，
            与label相同大小，bool类型）。
            当mode为'test'时，输入为image(-1, 3, -1, -1)返回pred (-1, 1, -1, -1)和
            logit (-1, num_classes, -1, -1) 通道维上代表每一类的概率值。
        output_stride (int): backbone 输出特征图相对于输入的下采样倍数，一般取值为8或16。
        aspp_with_sep_conv (bool): 在asspp模块是否采用separable convolutions。
        decoder_use_sep_conv (bool)： decoder模块是否采用separable convolutions。
        encoder_with_aspp (bool): 是否在encoder阶段采用aspp模块。
        enable_decoder (bool): 是否使用decoder模块。
        use_bce_loss (bool): 是否使用bce loss作为网络的损失函数，只能用于两类分割。可与dice loss同时使用。
        use_dice_loss (bool): 是否使用dice loss作为网络的损失函数，只能用于两类分割，可与bce loss同时使用。
            当use_bce_loss和use_dice_loss都为False时，使用交叉熵损失函数。
        class_weight (list/str): 交叉熵损失函数各类损失的权重。当class_weight为list的时候，长度应为
            num_classes。当class_weight为str时， weight.lower()应为'dynamic'，这时会根据每一轮各类像素的比重
            自行计算相应的权重，每一类的权重为：每类的比例 * num_classes。class_weight取默认值None是，各类的权重1，
            即平时使用的交叉熵损失函数。
        ignore_index (int): label上忽略的值，label为ignore_index的像素不参与损失函数的计算。
        fixed_input_shape (list): 长度为2，维度为1的list，如:[640,720]，用来固定模型输入:'image'的shape，默认为None。

    Raises:
        ValueError: use_bce_loss或use_dice_loss为真且num_calsses > 2。
        ValueError: class_weight为list, 但长度不等于num_class。
            class_weight为str, 但class_weight.low()不等于dynamic。
        TypeError: class_weight不为None时，其类型不是list或str。
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 input_channel=3,
                 mode='train',
                 output_stride=16,
                 aspp_with_sep_conv=True,
                 decoder_use_sep_conv=True,
                 encoder_with_aspp=True,
                 enable_decoder=True,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 fixed_input_shape=None,
                 pooling_stride=[1, 1],
                 pooling_crop_size=None,
                 aspp_with_se=False,
                 se_use_qsigmoid=False,
                 aspp_convs_filters=256,
                 aspp_with_concat_projection=True,
                 add_image_level_feature=True,
                 use_sum_merge=False,
                 conv_filters=256,
                 output_is_logits=False):
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
        self.backbone = backbone
        self.mode = mode
        self.use_bce_loss = use_bce_loss
        self.use_dice_loss = use_dice_loss
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.output_stride = output_stride
        self.aspp_with_sep_conv = aspp_with_sep_conv
        self.decoder_use_sep_conv = decoder_use_sep_conv
        self.encoder_with_aspp = encoder_with_aspp
        self.enable_decoder = enable_decoder
        self.fixed_input_shape = fixed_input_shape
        self.output_is_logits = output_is_logits
        self.aspp_convs_filters = aspp_convs_filters
        self.output_stride = output_stride
        self.pooling_crop_size = pooling_crop_size
        self.pooling_stride = pooling_stride
        self.se_use_qsigmoid = se_use_qsigmoid
        self.aspp_with_concat_projection = aspp_with_concat_projection
        self.add_image_level_feature = add_image_level_feature
        self.aspp_with_se = aspp_with_se
        self.use_sum_merge = use_sum_merge
        self.conv_filters = conv_filters

    def _encoder(self, input):
        # 编码器配置，采用ASPP架构，pooling + 1x1_conv + 三个不同尺度的空洞卷积并行, concat后1x1conv
        # ASPP_WITH_SEP_CONV：默认为真，使用depthwise可分离卷积，否则使用普通卷积
        # OUTPUT_STRIDE: 下采样倍数，8或16，决定aspp_ratios大小
        # aspp_ratios：ASPP模块空洞卷积的采样率

        if self.output_stride == 16:
            aspp_ratios = [6, 12, 18]
        elif self.output_stride == 8:
            aspp_ratios = [12, 24, 36]
        else:
            aspp_ratios = []

        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=None,
            initializer=fluid.initializer.TruncatedNormal(
                loc=0.0, scale=0.06))

        concat_logits = []
        with scope('encoder'):
            channel = self.aspp_convs_filters
            with scope("image_pool"):
                if self.pooling_crop_size is None:
                    image_avg = fluid.layers.reduce_mean(
                        input, [2, 3], keep_dim=True)
                else:
                    pool_w = int((self.pooling_crop_size[0] - 1.0) /
                                 self.output_stride + 1.0)
                    pool_h = int((self.pooling_crop_size[1] - 1.0) /
                                 self.output_stride + 1.0)
                    image_avg = fluid.layers.pool2d(
                        input,
                        pool_size=(pool_h, pool_w),
                        pool_stride=self.pooling_stride,
                        pool_type='avg',
                        pool_padding='VALID')

                act = qsigmoid if self.se_use_qsigmoid else bn_relu

                image_avg = act(
                    conv(
                        image_avg,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr))
                input_shape = fluid.layers.shape(input)
                image_avg = fluid.layers.resize_bilinear(image_avg,
                                                         input_shape[2:])
                if self.add_image_level_feature:
                    concat_logits.append(image_avg)

            with scope("aspp0"):
                aspp0 = bn_relu(
                    conv(
                        input,
                        channel,
                        1,
                        1,
                        groups=1,
                        padding=0,
                        param_attr=param_attr))
                concat_logits.append(aspp0)

            if aspp_ratios:
                with scope("aspp1"):
                    if self.aspp_with_sep_conv:
                        aspp1 = separate_conv(
                            input,
                            channel,
                            1,
                            3,
                            dilation=aspp_ratios[0],
                            act=relu)
                    else:
                        aspp1 = bn_relu(
                            conv(
                                input,
                                channel,
                                stride=1,
                                filter_size=3,
                                dilation=aspp_ratios[0],
                                padding=aspp_ratios[0],
                                param_attr=param_attr))
                    concat_logits.append(aspp1)
                with scope("aspp2"):
                    if self.aspp_with_sep_conv:
                        aspp2 = separate_conv(
                            input,
                            channel,
                            1,
                            3,
                            dilation=aspp_ratios[1],
                            act=relu)
                    else:
                        aspp2 = bn_relu(
                            conv(
                                input,
                                channel,
                                stride=1,
                                filter_size=3,
                                dilation=aspp_ratios[1],
                                padding=aspp_ratios[1],
                                param_attr=param_attr))
                    concat_logits.append(aspp2)
                with scope("aspp3"):
                    if self.aspp_with_sep_conv:
                        aspp3 = separate_conv(
                            input,
                            channel,
                            1,
                            3,
                            dilation=aspp_ratios[2],
                            act=relu)
                    else:
                        aspp3 = bn_relu(
                            conv(
                                input,
                                channel,
                                stride=1,
                                filter_size=3,
                                dilation=aspp_ratios[2],
                                padding=aspp_ratios[2],
                                param_attr=param_attr))
                    concat_logits.append(aspp3)

            with scope("concat"):
                data = fluid.layers.concat(concat_logits, axis=1)
                if self.aspp_with_concat_projection:
                    data = bn_relu(
                        conv(
                            data,
                            channel,
                            1,
                            1,
                            groups=1,
                            padding=0,
                            param_attr=param_attr))
                    data = fluid.layers.dropout(data, 0.9)
            if self.aspp_with_se:
                data = data * image_avg
            return data

    def _decoder_with_sum_merge(self, encode_data, decode_shortcut,
                                param_attr):
        decode_shortcut_shape = fluid.layers.shape(decode_shortcut)
        encode_data = fluid.layers.resize_bilinear(encode_data,
                                                   decode_shortcut_shape[2:])

        encode_data = conv(
            encode_data,
            self.conv_filters,
            1,
            1,
            groups=1,
            padding=0,
            param_attr=param_attr)

        with scope('merge'):
            decode_shortcut = conv(
                decode_shortcut,
                self.conv_filters,
                1,
                1,
                groups=1,
                padding=0,
                param_attr=param_attr)

            return encode_data + decode_shortcut

    def _decoder_with_concat(self, encode_data, decode_shortcut, param_attr):
        with scope('concat'):
            decode_shortcut = bn_relu(
                conv(
                    decode_shortcut,
                    48,
                    1,
                    1,
                    groups=1,
                    padding=0,
                    param_attr=param_attr))

            decode_shortcut_shape = fluid.layers.shape(decode_shortcut)
            encode_data = fluid.layers.resize_bilinear(
                encode_data, decode_shortcut_shape[2:])
            encode_data = fluid.layers.concat(
                [encode_data, decode_shortcut], axis=1)
        if self.decoder_use_sep_conv:
            with scope("separable_conv1"):
                encode_data = separate_conv(
                    encode_data, self.conv_filters, 1, 3, dilation=1, act=relu)
            with scope("separable_conv2"):
                encode_data = separate_conv(
                    encode_data, self.conv_filters, 1, 3, dilation=1, act=relu)
        else:
            with scope("decoder_conv1"):
                encode_data = bn_relu(
                    conv(
                        encode_data,
                        self.conv_filters,
                        stride=1,
                        filter_size=3,
                        dilation=1,
                        padding=1,
                        param_attr=param_attr))
            with scope("decoder_conv2"):
                encode_data = bn_relu(
                    conv(
                        encode_data,
                        self.conv_filters,
                        stride=1,
                        filter_size=3,
                        dilation=1,
                        padding=1,
                        param_attr=param_attr))
        return encode_data

    def _decoder(self, encode_data, decode_shortcut):
        # 解码器配置
        # encode_data：编码器输出
        # decode_shortcut: 从backbone引出的分支, resize后与encode_data concat
        # decoder_use_sep_conv: 默认为真，则concat后连接两个可分离卷积，否则为普通卷积
        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=None,
            initializer=fluid.initializer.TruncatedNormal(
                loc=0.0, scale=0.06))

        with scope('decoder'):
            if self.use_sum_merge:
                return self._decoder_with_sum_merge(
                    encode_data, decode_shortcut, param_attr)

            return self._decoder_with_concat(encode_data, decode_shortcut,
                                             param_attr)

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
        return inputs

    def build_net(self, inputs):
        # 在两类分割情况下，当loss函数选择dice_loss或bce_loss的时候，最后logit输出通道数设置为1
        if self.use_dice_loss or self.use_bce_loss:
            self.num_classes = 1
        image = inputs['image']

        if 'MobileNetV3' in self.backbone.__class__.__name__:
            data, decode_shortcut = self.backbone(image)
        else:
            data, decode_shortcuts = self.backbone(image)
            decode_shortcut = decode_shortcuts[self.backbone.decode_points]

        # 编码器解码器设置
        if self.encoder_with_aspp:
            data = self._encoder(data)
        if self.enable_decoder:
            data = self._decoder(data, decode_shortcut)

        # 根据类别数设置最后一个卷积层输出，并resize到图片原始尺寸
        param_attr = fluid.ParamAttr(
            name=name_scope + 'weights',
            regularizer=fluid.regularizer.L2DecayRegularizer(
                regularization_coeff=0.0),
            initializer=fluid.initializer.TruncatedNormal(
                loc=0.0, scale=0.01))
        if not self.output_is_logits:
            with scope('logit'):
                with fluid.name_scope('last_conv'):
                    logit = conv(
                        data,
                        self.num_classes,
                        1,
                        stride=1,
                        padding=0,
                        bias_attr=True,
                        param_attr=param_attr)
        else:
            logit = data

        image_shape = fluid.layers.shape(image)
        logit = fluid.layers.resize_bilinear(logit, image_shape[2:])

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

        return logit
