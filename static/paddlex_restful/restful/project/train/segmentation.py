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
    from paddlex.seg import transforms
    seg_list = []
    min_value = max(params.image_shape) * 4 // 5
    max_value = max(params.image_shape) * 6 // 5
    seg_list.extend([
        transforms.ResizeRangeScaling(
            min_value=min_value, max_value=max_value),
        transforms.RandomBlur(prob=params.blur_prob)
    ])
    if params.rotate:
        seg_list.append(
            transforms.RandomRotate(rotate_range=params.max_rotation))
    if params.scale_aspect:
        seg_list.append(
            transforms.RandomScaleAspect(
                min_scale=params.min_ratio, aspect_ratio=params.aspect_ratio))
    seg_list.extend([
        transforms.RandomDistort(
            brightness_range=params.brightness_range,
            brightness_prob=params.brightness_prob,
            contrast_range=params.contrast_range,
            contrast_prob=params.contrast_prob,
            saturation_range=params.saturation_range,
            saturation_prob=params.saturation_prob,
            hue_range=params.hue_range,
            hue_prob=params.hue_prob),
        transforms.RandomVerticalFlip(prob=params.vertical_flip_prob),
        transforms.RandomHorizontalFlip(prob=params.horizontal_flip_prob),
        transforms.RandomPaddingCrop(crop_size=max(params.image_shape)),
        transforms.Normalize(
            mean=params.image_mean, std=params.image_std)
    ])

    train_transforms = transforms.Compose(seg_list)
    eval_transforms = transforms.Compose([
        transforms.ResizeByLong(long_size=max(params.image_shape)),
        transforms.Padding(target_size=max(params.image_shape)),
        transforms.Normalize(
            mean=params.image_mean, std=params.image_std)
    ])
    return train_transforms, eval_transforms


def build_datasets(dataset_path, train_transforms, eval_transforms):
    import paddlex as pdx
    train_file_list = osp.join(dataset_path, 'train_list.txt')
    eval_file_list = osp.join(dataset_path, 'val_list.txt')
    label_list = osp.join(dataset_path, 'labels.txt')
    train_dataset = pdx.datasets.SegDataset(
        data_dir=dataset_path,
        file_list=train_file_list,
        label_list=label_list,
        transforms=train_transforms,
        shuffle=True)
    eval_dataset = pdx.datasets.SegDataset(
        data_dir=dataset_path,
        file_list=eval_file_list,
        label_list=label_list,
        transforms=eval_transforms)
    return train_dataset, eval_dataset


def build_optimizer(step_each_epoch, params):
    import paddle.fluid as fluid
    if params.lr_policy == 'Piecewise':
        gamma = 0.1
        bd = [step_each_epoch * e for e in params.lr_decay_epochs]
        lr = [params.learning_rate * (gamma**i) for i in range(len(bd) + 1)]
        decayed_lr = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    elif params.lr_policy == 'Polynomial':
        decay_step = params.num_epochs * step_each_epoch
        decayed_lr = fluid.layers.polynomial_decay(
            params.learning_rate, decay_step, end_learning_rate=0, power=0.9)
    elif params.lr_policy == 'Cosine':
        decayed_lr = fluid.layers.cosine_decay(
            params.learning_rate, step_each_epoch, params.num_epochs)
    else:
        raise Exception(
            'lr_policy only support Polynomial or Piecewise, but you set {}'.
            format(params.lr_policy))

    if params.optimizer.lower() == 'sgd':
        momentum = 0.9
        regularize_coef = 1e-4
        optimizer = fluid.optimizer.Momentum(
            learning_rate=decayed_lr,
            momentum=momentum,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=regularize_coef), )
    elif params.optimizer.lower() == 'adam':
        momentum = 0.9
        momentum2 = 0.999
        regularize_coef = 1e-4
        optimizer = fluid.optimizer.Adam(
            learning_rate=decayed_lr,
            beta1=momentum,
            beta2=momentum2,
            regularization=fluid.regularizer.L2Decay(
                regularization_coeff=regularize_coef), )

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
    segmenter = getattr(pdx.cv.models, 'HRNet'
                        if params.model.startswith('HRNet') else params.model)
    use_dice_loss, use_bce_loss = params.loss_type
    backbone = params.backbone
    sensitivities_path = params.sensitivities_path
    eval_metric_loss = params.eval_metric_loss
    if eval_metric_loss is None:
        eval_metric_loss = 0.05
    if params.model in ['UNet', 'HRNet_W18', 'FastSCNN']:
        model = segmenter(
            num_classes=len(train_dataset.labels),
            use_bce_loss=use_bce_loss,
            use_dice_loss=use_dice_loss)
    elif params.model == 'DeepLabv3p':
        model = segmenter(
            num_classes=len(train_dataset.labels),
            backbone=backbone,
            use_bce_loss=use_bce_loss,
            use_dice_loss=use_dice_loss)
        if backbone == 'MobileNetV3_large_x1_0_ssld':
            model.pooling_crop_size = params.image_shape
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
