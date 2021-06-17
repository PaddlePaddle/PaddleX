# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from . import cv
from .cv.models.utils.visualize import visualize_segmentation
import paddlex
import paddlex.utils.logging as logging

transforms = paddlex.cv.transforms

visualize = visualize_segmentation


class UNet(cv.models.UNet):
    def __init__(self,
                 num_classes=2,
                 upsample_mode='bilinear',
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=None,
                 input_channel=None):
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classification"
            )
        elif num_classes == 2:
            if use_bce_loss and use_dice_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1), ('DiceLoss', 1)]
            elif use_bce_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1)]
            elif use_dice_loss:
                use_mixed_loss = [('DiceLoss', 1)]
            else:
                use_mixed_loss = False
        else:
            use_mixed_loss = False

        if class_weight is not None:
            logging.warning(
                "`class_weight` is not supported in PaddleX 2.0 currently and is forcibly set to None."
            )
        if ignore_index is not None:
            logging.warning(
                "`ignore_index` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 255."
            )
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )

        if upsample_mode == 'bilinear':
            use_deconv = False
        else:
            use_deconv = True
        super(UNet, self).__init__(
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            use_deconv=use_deconv)


class DeepLabv3p(cv.models.DeepLabV3P):
    def __init__(self,
                 num_classes=2,
                 backbone='ResNet50_vd',
                 output_stride=8,
                 aspp_with_sep_conv=None,
                 decoder_use_sep_conv=None,
                 encoder_with_aspp=None,
                 enable_decoder=None,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=None,
                 pooling_crop_size=None,
                 input_channel=None):
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classification"
            )
        elif num_classes == 2:
            if use_bce_loss and use_dice_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1), ('DiceLoss', 1)]
            elif use_bce_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1)]
            elif use_dice_loss:
                use_mixed_loss = [('DiceLoss', 1)]
            else:
                use_mixed_loss = False
        else:
            use_mixed_loss = False

        if aspp_with_sep_conv is not None:
            logging.warning(
                "`aspp_with_sep_conv` is deprecated in PaddleX 2.0 and will not take effect. "
                "Defaults to True")
        if decoder_use_sep_conv is not None:
            logging.warning(
                "`decoder_use_sep_conv` is deprecated in PaddleX 2.0 and will not take effect. "
                "Defaults to True")
        if encoder_with_aspp is not None:
            logging.warning(
                "`encoder_with_aspp` is deprecated in PaddleX 2.0 and will not take effect. "
                "Defaults to True")
        if enable_decoder is not None:
            logging.warning(
                "`enable_decoder` is deprecated in PaddleX 2.0 and will not take effect. "
                "Defaults to True")
        if class_weight is not None:
            logging.warning(
                "`class_weight` is not supported in PaddleX 2.0 currently and is forcibly set to None."
            )
        if ignore_index is not None:
            logging.warning(
                "`ignore_index` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 255."
            )
        if pooling_crop_size is not None:
            logging.warning(
                "Backbone 'MobileNetV3_large_x1_0_ssld' is currently not supported in PaddleX 2.0. "
                "`pooling_crop_size` will not take effect. Defaults to None")
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )

        super(DeepLabv3p, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            use_mixed_loss=use_mixed_loss,
            output_stride=output_stride)


class HRNet(cv.models.HRNet):
    def __init__(self,
                 num_classes=2,
                 width=18,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=None,
                 input_channel=None):
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classification"
            )
        elif num_classes == 2:
            if use_bce_loss and use_dice_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1), ('DiceLoss', 1)]
            elif use_bce_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1)]
            elif use_dice_loss:
                use_mixed_loss = [('DiceLoss', 1)]
            else:
                use_mixed_loss = False
        else:
            use_mixed_loss = False

        if class_weight is not None:
            logging.warning(
                "`class_weight` is not supported in PaddleX 2.0 currently and is forcibly set to None."
            )
        if ignore_index is not None:
            logging.warning(
                "`ignore_index` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 255."
            )
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )

        super(HRNet, self).__init__(
            num_classes=num_classes,
            width=width,
            use_mixed_loss=use_mixed_loss)


class FastSCNN(cv.models.FastSCNN):
    def __init__(self,
                 num_classes=2,
                 use_bce_loss=False,
                 use_dice_loss=False,
                 class_weight=None,
                 ignore_index=255,
                 multi_loss_weight=None,
                 input_channel=3):
        if num_classes > 2 and (use_bce_loss or use_dice_loss):
            raise ValueError(
                "dice loss and bce loss is only applicable to binary classification"
            )
        elif num_classes == 2:
            if use_bce_loss and use_dice_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1), ('DiceLoss', 1)]
            elif use_bce_loss:
                use_mixed_loss = [('CrossEntropyLoss', 1)]
            elif use_dice_loss:
                use_mixed_loss = [('DiceLoss', 1)]
            else:
                use_mixed_loss = False
        else:
            use_mixed_loss = False

        if class_weight is not None:
            logging.warning(
                "`class_weight` is not supported in PaddleX 2.0 currently and is forcibly set to None."
            )
        if ignore_index is not None:
            logging.warning(
                "`ignore_index` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 255."
            )
        if multi_loss_weight is not None:
            logging.warning(
                "`multi_loss_weight` is deprecated in PaddleX 2.0 and will not take effect. "
                "Defaults to [1.0, 0.4]")
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )

        super(FastSCNN, self).__init__(
            num_classes=num_classes, use_mixed_loss=use_mixed_loss)
