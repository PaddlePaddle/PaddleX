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


def build_transforms(params):
    from paddlex.cls import transforms
    crop_size = params.image_shape[0]
    train_transforms = transforms.Compose([
        transforms.RandomCrop(
            crop_size=crop_size,
            lower_scale=0.88,
            lower_ratio=3. / 4,
            upper_ratio=4. / 3),
        transforms.RandomHorizontalFlip(prob=params.horizontal_flip_prob),
        transforms.RandomVerticalFlip(prob=params.vertical_flip_prob),
        transforms.RandomDistort(
            brightness_range=params.brightness_range,
            brightness_prob=params.brightness_prob,
            contrast_range=params.contrast_range,
            contrast_prob=params.contrast_prob,
            saturation_range=params.saturation_range,
            saturation_prob=params.saturation_prob,
            hue_range=params.hue_range,
            hue_prob=params.hue_prob), transforms.RandomRotate(
                rotate_range=params.rotate_range,
                prob=params.rotate_prob), transforms.Normalize(
                    mean=params.image_mean, std=params.image_std)
    ])
    eval_transforms = transforms.Compose([
        transforms.ResizeByShort(short_size=int(crop_size * 1.143)),
        transforms.CenterCrop(crop_size=crop_size), transforms.Normalize(
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


def build_optimizer(step_each_epoch, params):
    import paddle.fluid as fluid
    from paddle.fluid.regularizer import L2Decay
    learning_rate = params.learning_rate
    num_epochs = params.num_epochs
    if params.lr_policy == 'Cosine':
        learning_rate = fluid.layers.cosine_decay(
            learning_rate=learning_rate,
            step_each_epoch=step_each_epoch,
            epochs=num_epochs)
    elif params.lr_policy == 'Linear':
        learning_rate = fluid.layers.polynomial_decay(
            learning_rate=learning_rate,
            decay_steps=step_each_epoch * num_epochs,
            end_learning_rate=0.0,
            power=1.0)
    elif params.lr_policy == 'Piecewise':
        lr_decay_epochs = params.lr_decay_epochs
        values = [
            learning_rate * (0.1**i) for i in range(len(lr_decay_epochs) + 1)
        ]
        boundaries = [b * step_each_epoch for b in lr_decay_epochs]
        learning_rate = fluid.layers.piecewise_decay(
            boundaries=boundaries, values=values)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        regularization=L2Decay(1e-04))
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

    optimizer = build_optimizer(step_each_epoch, params)
    classifier = getattr(pdx.cv.models, params.model)
    sensitivities_path = params.sensitivities_path
    eval_metric_loss = params.eval_metric_loss
    if eval_metric_loss is None:
        eval_metric_loss = 0.05
    model = classifier(num_classes=len(train_dataset.labels))
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
        sensitivities_file=sensitivities_path,
        eval_metric_loss=eval_metric_loss,
        resume_checkpoint=params.resume_checkpoint)
