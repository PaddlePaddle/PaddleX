# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
    seg_list = []
    seg_list.extend([
        T.Resize(target_size=params.image_shape),
        T.RandomBlur(prob=params.blur_prob)
    ])
    if params.scale_aspect:
        seg_list.append(
            T.RandomScaleAspect(
                min_scale=params.min_ratio, aspect_ratio=params.aspect_ratio))
    seg_list.extend([
        T.RandomDistort(
            brightness_range=params.brightness_range,
            brightness_prob=params.brightness_prob,
            contrast_range=params.contrast_range,
            contrast_prob=params.contrast_prob,
            saturation_range=params.saturation_range,
            saturation_prob=params.saturation_prob,
            hue_range=params.hue_range,
            hue_prob=params.hue_prob),
        T.RandomVerticalFlip(prob=params.vertical_flip_prob),
        T.RandomHorizontalFlip(prob=params.horizontal_flip_prob), T.Normalize(
            mean=params.image_mean, std=params.image_std)
    ])

    train_transforms = T.Compose(seg_list)
    eval_transforms = T.Compose([
        T.Resize(target_size=params.image_shape), T.Normalize(
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


def build_optimizer(parameters, step_each_epoch, params):
    import paddle
    from paddle.regularizer import L2Decay
    learning_rate = params.learning_rate
    num_epochs = params.num_epochs
    if params.lr_policy == 'Piecewise':
        lr_decay_epochs = params.lr_decay_epochs
        gamma = 0.1
        boundaries = [step_each_epoch * e for e in lr_decay_epochs]
        values = [
            learning_rate * (gamma**i)
            for i in range(len(lr_decay_epochs) + 1)
        ]
        decayed_lr = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values)
    elif params.lr_policy == 'Polynomial':
        decay_step = num_epochs * step_each_epoch
        decayed_lr = paddle.optimizer.lr.PolynomialDecay(
            learning_rate=learning_rate,
            decay_steps=decay_step,
            end_lr=0.0,
            power=.9)
    elif params.lr_policy == 'Cosine':
        decayed_lr = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=.001, T_max=step_each_epoch * num_epochs)
    else:
        raise Exception(
            'lr_policy only support Polynomial or Piecewise, but you set {}'.
            format(params.lr_policy))

    if params.optimizer.lower() == 'sgd':
        momentum = 0.9
        regularize_coef = 1e-4
        optimizer = paddle.optimizer.Momentum(
            learning_rate=decayed_lr,
            momentum=momentum,
            weight_decay=L2Decay(regularize_coef),
            parameters=parameters)
    elif params.optimizer.lower() == 'adam':
        momentum = 0.9
        momentum2 = 0.999
        regularize_coef = 1e-4
        optimizer = paddle.optimizer.Adam(
            learning_rate=decayed_lr,
            beta1=momentum,
            beta2=momentum2,
            weight_decay=L2Decay(regularize_coef),
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

    segmenter = getattr(pdx.seg, 'HRNet'
                        if params.model.startswith('HRNet') else params.model)
    use_dice_loss, use_bce_loss = params.loss_type
    if use_bce_loss and use_dice_loss:
        use_mixed_loss = [('CrossEntropyLoss', 1), ('DiceLoss', 1)]
    elif use_bce_loss:
        use_mixed_loss = [('CrossEntropyLoss', 1)]
    elif use_dice_loss:
        use_mixed_loss = [('DiceLoss', 1)]
    else:
        use_mixed_loss = False

    backbone = params.backbone
    sensitivities_path = params.sensitivities_path
    pruned_flops = params.pruned_flops
    if params.model in ['UNet', 'HRNet_W18', 'FastSCNN', 'BiSeNetV2']:
        model = segmenter(
            num_classes=len(train_dataset.labels),
            use_mixed_loss=use_mixed_loss)
    elif params.model == 'DeepLabV3P':
        model = segmenter(
            num_classes=len(train_dataset.labels),
            backbone=backbone,
            use_mixed_loss=use_mixed_loss)

    if sensitivities_path is not None:
        # load weights
        model.net_initialize(pretrain_weights=osp.join(pretrain_weights,
                                                       'model.pdparams'))
        pretrain_weights = None
        # prune
        dataset = eval_dataset or train_dataset
        inputs = [1, 3] + list(dataset[0]['image'].shape[:2])
        model.pruner = L1NormFilterPruner(
            model.net, inputs=inputs, sen_file=sensitivities_path)
        #model.pruner.sensitive_prune(pruned_flops=pruned_flops)
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
