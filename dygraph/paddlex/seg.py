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

import os.path as osp
from paddleslim import L1NormFilterPruner
from . import cv
from .cv.models.utils.visualize import visualize_segmentation
from paddlex.cv.transforms import seg_transforms
import paddlex.utils.logging as logging
from paddlex.utils.checkpoint import seg_pretrain_weights_dict

transforms = seg_transforms

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

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='COCO',
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              use_vdl=False,
              sensitivities_file=None,
              pruned_flops=.2,
              early_stop=False,
              early_stop_patience=5):
        _legacy_train(
            self,
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


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

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              use_vdl=False,
              sensitivities_file=None,
              pruned_flops=.2,
              early_stop=False,
              early_stop_patience=5):
        _legacy_train(
            self,
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


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

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              use_vdl=False,
              sensitivities_file=None,
              pruned_flops=.2,
              early_stop=False,
              early_stop_patience=5):
        _legacy_train(
            self,
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


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

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=2,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='CITYSCAPES',
              optimizer=None,
              learning_rate=0.01,
              lr_decay_power=0.9,
              use_vdl=False,
              sensitivities_file=None,
              pruned_flops=.2,
              early_stop=False,
              early_stop_patience=5):
        _legacy_train(
            self,
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            optimizer=optimizer,
            learning_rate=learning_rate,
            lr_decay_power=lr_decay_power,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


def _legacy_train(model, num_epochs, train_dataset, train_batch_size,
                  eval_dataset, save_interval_epochs, log_interval_steps,
                  save_dir, pretrain_weights, optimizer, learning_rate,
                  lr_decay_power, use_vdl, sensitivities_file, pruned_flops,
                  early_stop, early_stop_patience):
    model.labels = train_dataset.labels
    if model.losses is None:
        model.losses = model.default_loss()

    # initiate weights
    if pretrain_weights is not None and not osp.exists(pretrain_weights):
        if pretrain_weights not in seg_pretrain_weights_dict[model.model_name]:
            logging.warning("Path of pretrain_weights('{}') does not exist!".
                            format(pretrain_weights))
            logging.warning("Pretrain_weights is forcibly set to '{}'. "
                            "If don't want to use pretrain weights, "
                            "set pretrain_weights to be None.".format(
                                seg_pretrain_weights_dict[model.model_name][
                                    0]))
            pretrain_weights = seg_pretrain_weights_dict[model.model_name][0]
    pretrained_dir = osp.join(save_dir, 'pretrain')
    model.net_initialize(
        pretrain_weights=pretrain_weights, save_dir=pretrained_dir)

    if sensitivities_file is not None:
        dataset = eval_dataset or train_dataset
        inputs = [1, 3] + list(dataset[0]['image'].shape[:2])
        model.pruner = L1NormFilterPruner(
            model.net, inputs=inputs, sen_file=sensitivities_file)
        model.pruner.sensitive_prune(pruned_flops=pruned_flops)

    # build optimizer if not defined
    if optimizer is None:
        num_steps_each_epoch = train_dataset.num_samples // train_batch_size
        model.optimizer = model.default_optimizer(
            model.net.parameters(), learning_rate, num_epochs,
            num_steps_each_epoch, lr_decay_power)
    else:
        model.optimizer = optimizer

    model.train_loop(
        num_epochs=num_epochs,
        train_dataset=train_dataset,
        train_batch_size=train_batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=save_interval_epochs,
        log_interval_steps=log_interval_steps,
        save_dir=save_dir,
        early_stop=early_stop,
        early_stop_patience=early_stop_patience,
        use_vdl=use_vdl)
