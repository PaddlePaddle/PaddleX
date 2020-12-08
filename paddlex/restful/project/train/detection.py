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
import math
import pickle
import os


def build_yolo_transforms(params):
    from paddlex.det import transforms
    target_size = params.image_shape[0]
    use_mixup = params.use_mixup
    dt_list = []
    if use_mixup:
        dt_list.append(
            transforms.MixupImage(
                alpha=params.mixup_alpha,
                beta=params.mixup_beta,
                mixup_epoch=int(params.num_epochs * 25. / 27)))
    dt_list.extend([
        transforms.RandomDistort(
            brightness_range=params.brightness_range,
            brightness_prob=params.brightness_prob,
            contrast_range=params.contrast_range,
            contrast_prob=params.contrast_prob,
            saturation_range=params.saturation_range,
            saturation_prob=params.saturation_prob,
            hue_range=params.hue_range,
            hue_prob=params.hue_prob), transforms.RandomExpand(
                prob=params.expand_prob,
                fill_value=[float(int(x * 255)) for x in params.image_mean])
    ])
    crop_image = params.crop_image
    if crop_image:
        dt_list.append(transforms.RandomCrop())
    dt_list.extend([
        transforms.Resize(
            target_size=target_size, interp='RANDOM'),
        transforms.RandomHorizontalFlip(prob=params.horizontal_flip_prob),
        transforms.Normalize(
            mean=params.image_mean, std=params.image_std)
    ])
    train_transforms = transforms.Compose(dt_list)
    eval_transforms = transforms.Compose([
        transforms.Resize(
            target_size=target_size, interp='CUBIC'),
        transforms.Normalize(
            mean=params.image_mean, std=params.image_std),
    ])
    return train_transforms, eval_transforms


def build_rcnn_transforms(params):
    from paddlex.det import transforms
    short_size = min(params.image_shape)
    max_size = max(params.image_shape)
    train_transforms = transforms.Compose([
        transforms.RandomDistort(
            brightness_range=params.brightness_range,
            brightness_prob=params.brightness_prob,
            contrast_range=params.contrast_range,
            contrast_prob=params.contrast_prob,
            saturation_range=params.saturation_range,
            saturation_prob=params.saturation_prob,
            hue_range=params.hue_range,
            hue_prob=params.hue_prob),
        transforms.RandomHorizontalFlip(prob=params.horizontal_flip_prob),
        transforms.Normalize(
            mean=params.image_mean, std=params.image_std),
        transforms.ResizeByShort(
            short_size=short_size, max_size=max_size),
        transforms.Padding(coarsest_stride=32 if params.with_fpn else 1),
    ])
    eval_transforms = transforms.Compose([
        transforms.Normalize(), transforms.ResizeByShort(
            short_size=short_size, max_size=max_size),
        transforms.Padding(coarsest_stride=32 if params.with_fpn else 1)
    ])
    return train_transforms, eval_transforms


def build_voc_datasets(dataset_path, train_transforms, eval_transforms):
    import paddlex as pdx
    train_file_list = osp.join(dataset_path, 'train_list.txt')
    eval_file_list = osp.join(dataset_path, 'val_list.txt')
    label_list = osp.join(dataset_path, 'labels.txt')
    train_dataset = pdx.datasets.VOCDetection(
        data_dir=dataset_path,
        file_list=train_file_list,
        label_list=label_list,
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.VOCDetection(
        data_dir=dataset_path,
        file_list=eval_file_list,
        label_list=label_list,
        transforms=eval_transforms)
    return train_dataset, eval_dataset


def build_coco_datasets(dataset_path, train_transforms, eval_transforms):
    import paddlex as pdx
    data_dir = osp.join(dataset_path, 'JPEGImages')
    train_ann_file = osp.join(dataset_path, 'train.json')
    eval_ann_file = osp.join(dataset_path, 'val.json')
    train_dataset = pdx.datasets.CocoDetection(
        data_dir=data_dir,
        ann_file=train_ann_file,
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.CocoDetection(
        data_dir=data_dir, ann_file=eval_ann_file, transforms=eval_transforms)
    return train_dataset, eval_dataset


def build_optimizer(step_each_epoch, params):
    import paddle.fluid as fluid
    from paddle.fluid.regularizer import L2Decay
    learning_rate = params.learning_rate
    num_epochs = params.num_epochs
    lr_decay_epochs = params.lr_decay_epochs
    warmup_steps = params.warmup_steps
    warmup_start_lr = params.warmup_start_lr

    boundaries = [b * step_each_epoch for b in lr_decay_epochs]
    values = [
        learning_rate * (0.1**i) for i in range(len(lr_decay_epochs) + 1)
    ]
    lr = fluid.layers.piecewise_decay(boundaries=boundaries, values=values)
    lr = fluid.layers.linear_lr_warmup(
        learning_rate=lr,
        warmup_steps=warmup_steps,
        start_lr=warmup_start_lr,
        end_lr=learning_rate)
    factor = 1e-04 if params.model in ['FasterRCNN', 'MaskRCNN'] else 5e-04
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr, momentum=0.9, regularization=L2Decay(factor))
    return optimizer


def train(task_path, dataset_path, params):
    import paddlex as pdx
    pdx.log_level = 3
    if params.model in ['YOLOv3', 'PPYOLO']:
        train_transforms, eval_transforms = build_yolo_transforms(params)
    elif params.model in ['FasterRCNN', 'MaskRCNN']:
        train_transforms, eval_transforms = build_rcnn_transforms(params)
    if osp.exists(osp.join(dataset_path, 'JPEGImages')) and \
        osp.exists(osp.join(dataset_path, 'train.json')) and \
        osp.exists(osp.join(dataset_path, 'val.json')):
        train_dataset, eval_dataset = build_coco_datasets(
            dataset_path=dataset_path,
            train_transforms=train_transforms,
            eval_transforms=eval_transforms)
    elif osp.exists(osp.join(dataset_path, 'train_list.txt')) and \
        osp.exists(osp.join(dataset_path, 'val_list.txt')) and \
        osp.exists(osp.join(dataset_path, 'labels.txt')):
        train_dataset, eval_dataset = build_voc_datasets(
            dataset_path=dataset_path,
            train_transforms=train_transforms,
            eval_transforms=eval_transforms)
    step_each_epoch = train_dataset.num_samples // params.batch_size
    train_batch_size = params.batch_size
    save_interval_epochs = params.save_interval_epochs
    save_dir = osp.join(task_path, 'output')
    pretrain_weights = params.pretrain_weights

    optimizer = build_optimizer(step_each_epoch, params)
    detector = getattr(pdx.cv.models, params.model)
    num_classes = len(train_dataset.labels) if params.model in ['YOLOv3', 'PPYOLO'] else \
        len(train_dataset.labels) + 1
    sensitivities_path = params.sensitivities_path
    eval_metric_loss = params.eval_metric_loss
    if eval_metric_loss is None:
        eval_metric_loss = 0.05
    model = detector(num_classes=num_classes, backbone=params.backbone)
    if params.model in ['YOLOv3', 'PPYOLO']:
        model.train_random_shapes = params.random_shape_sizes
    if params.model == 'YOLOv3':
        model.train(
            num_epochs=params.num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
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
    else:
        model.train(
            num_epochs=params.num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=2,
            save_dir=save_dir,
            pretrain_weights=pretrain_weights,
            optimizer=optimizer,
            use_vdl=True,
            resume_checkpoint=params.resume_checkpoint)
