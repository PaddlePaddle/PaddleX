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
from paddlex.cv.transforms import cls_transforms
import paddlex.utils.logging as logging

transforms = cls_transforms


class ResNet18(cv.models.ResNet18):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet18, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet34(cv.models.ResNet34):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet34, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet50(cv.models.ResNet50):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet50, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet101(cv.models.ResNet101):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet101, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet50_vd(cv.models.ResNet50_vd):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet50_vd, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet101_vd(cv.models.ResNet101_vd):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet101_vd, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet50_vd_ssld(cv.models.ResNet50_vd_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet50_vd_ssld, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ResNet101_vd_ssld(cv.models.ResNet101_vd_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet101_vd_ssld, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class DarkNet53(cv.models.DarkNet53):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DarkNet53, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class MobileNetV1(cv.models.MobileNetV1):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV1, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class MobileNetV2(cv.models.MobileNetV2):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV2, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class MobileNetV3_small(cv.models.MobileNetV3_small):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_small, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class MobileNetV3_large(cv.models.MobileNetV3_large):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_large, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class MobileNetV3_small_ssld(cv.models.MobileNetV3_small_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_small_ssld, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class MobileNetV3_large_ssld(cv.models.MobileNetV3_large_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_large_ssld, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class Xception41(cv.models.Xception41):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(Xception41, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class Xception65(cv.models.Xception65):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(Xception65, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class DenseNet121(cv.models.DenseNet121):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DenseNet121, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class DenseNet161(cv.models.DenseNet161):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DenseNet161, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class DenseNet201(cv.models.DenseNet201):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DenseNet201, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class ShuffleNetV2(cv.models.ShuffleNetV2):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ShuffleNetV2, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class HRNet_W18(cv.models.HRNet_W18_C):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(HRNet_W18, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


class AlexNet(cv.models.AlexNet):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(AlexNet, self).__init__(num_classes=num_classes)

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              save_interval_epochs=1,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=0.025,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=[30, 60, 90],
              lr_decay_gamma=0.1,
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
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            use_vdl=use_vdl,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)


def _legacy_train(model, num_epochs, train_dataset, train_batch_size,
                  eval_dataset, save_interval_epochs, log_interval_steps,
                  save_dir, pretrain_weights, optimizer, learning_rate,
                  warmup_steps, warmup_start_lr, lr_decay_epochs,
                  lr_decay_gamma, use_vdl, sensitivities_file, pruned_flops,
                  early_stop, early_stop_patience):
    model.labels = train_dataset.labels

    # initiate weights
    if pretrain_weights is not None and not osp.exists(pretrain_weights):
        if pretrain_weights not in ['IMAGENET']:
            logging.warning("Path of pretrain_weights('{}') does not exist!".
                            format(pretrain_weights))
            logging.warning("Pretrain_weights is forcibly set to 'IMAGENET'. "
                            "If don't want to use pretrain weights, "
                            "set pretrain_weights to be None.")
            pretrain_weights = 'IMAGENET'
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
        num_steps_each_epoch = len(train_dataset) // train_batch_size
        model.optimizer = model.default_optimizer(
            parameters=model.net.parameters(),
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            warmup_start_lr=warmup_start_lr,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_gamma=lr_decay_gamma,
            num_steps_each_epoch=num_steps_each_epoch)
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
