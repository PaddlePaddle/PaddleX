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

import os.path as osp
from paddleslim import L1NormFilterPruner


def build_transforms(params):
    from paddlex import transforms as T
    crop_size = params.image_shape[0]
    train_transforms = T.Compose([
        T.RandomCrop(
            crop_size=crop_size,
            scaling=[.88, 1.],
            aspect_ratio=[3. / 4, 4. / 3]),
        T.RandomHorizontalFlip(prob=params.horizontal_flip_prob),
        T.RandomVerticalFlip(prob=params.vertical_flip_prob), T.RandomDistort(
            brightness_range=params.brightness_range,
            brightness_prob=params.brightness_prob,
            contrast_range=params.contrast_range,
            contrast_prob=params.contrast_prob,
            saturation_range=params.saturation_range,
            saturation_prob=params.saturation_prob,
            hue_range=params.hue_range,
            hue_prob=params.hue_prob), T.Normalize(
                mean=params.image_mean, std=params.image_std)
    ])
    eval_transforms = T.Compose([
        T.ResizeByShort(short_size=int(crop_size * 1.143)),
        T.CenterCrop(crop_size=crop_size), T.Normalize(
            mean=params.image_mean, std=params.image_std)
    ])
    return train_transforms, eval_transforms


def build_datasets(dataset_path, train_transforms, eval_transforms):
    import paddlex as pdx
    train_file_list = osp.join(dataset_path, 'train_list.txt')
    eval_file_list = osp.join(dataset_path, 'val_list.txt')
    label_list = osp.join(dataset_path, 'labels.txt')
    train_dataset = pdx.datasets.ImageNet(
        data_dir=dataset_path,
        file_list=train_file_list,
        label_list=label_list,
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.ImageNet(
        data_dir=dataset_path,
        file_list=eval_file_list,
        label_list=label_list,
        transforms=eval_transforms)
    return train_dataset, eval_dataset


def build_optimizer(parameters, step_each_epoch, params):
    import paddle
    from paddle.regularizer import L2Decay
    learning_rate = params.learning_rate
    num_epochs = params.num_epochs
    if params.lr_policy == 'Cosine':
        learning_rate = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=.001, T_max=step_each_epoch * num_epochs)
    elif params.lr_policy == 'Linear':
        learning_rate = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=learning_rate,
            decay_steps=step_each_epoch * num_epochs,
            end_lr=0.0,
            power=1.0)
    elif params.lr_policy == 'Piecewise':
        lr_decay_epochs = params.lr_decay_epochs
        gamma = 0.1
        boundaries = [step_each_epoch * e for e in lr_decay_epochs]
        values = [
            learning_rate * (gamma**i)
            for i in range(len(lr_decay_epochs) + 1)
        ]
        learning_rate = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values)
    optimizer = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=.9,
        weight_decay=L2Decay(1e-04),
        parameters=parameters)
    return optimizer


def train(task_path, dataset_path, params):
    import paddlex as pdx
    pdx.log_level = 3
    train_transforms, eval_transforms = build_transforms(params)
    train_dataset, eval_dataset = build_datasets(
        dataset_path=dataset_path,
        train_transforms=train_transforms,
        eval_transforms=eval_transforms)

    step_each_epoch = train_dataset.num_samples // params.batch_size
    save_interval_epochs = params.save_interval_epochs
    save_dir = osp.join(task_path, 'output')
    pretrain_weights = params.pretrain_weights
    if pretrain_weights is not None and osp.exists(pretrain_weights):
        pretrain_weights = osp.join(pretrain_weights, 'model.pdparams')

    classifier = getattr(pdx.cls, params.model)
    sensitivities_path = params.sensitivities_path
    pruned_flops = params.pruned_flops
    model = classifier(num_classes=len(train_dataset.labels))
    if sensitivities_path is not None:
        # load weights
        model.net_initialize(pretrain_weights=pretrain_weights)
        pretrain_weights = None
        # prune
        inputs = [1, 3] + list(eval_dataset[0]['image'].shape[:2])
        model.pruner = L1NormFilterPruner(
            model.net, inputs=inputs, sen_file=sensitivities_path)
        model.prune(pruned_flops=pruned_flops)

    optimizer = build_optimizer(model.net.parameters(), step_each_epoch,
                                params)
    model.train(
        num_epochs=params.num_epochs,
        train_dataset=train_dataset,
        train_batch_size=params.batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=save_interval_epochs,
        log_interval_steps=2,
        save_dir=save_dir,
        pretrain_weights=pretrain_weights,
        optimizer=optimizer,
        use_vdl=True,
        resume_checkpoint=params.resume_checkpoint)
