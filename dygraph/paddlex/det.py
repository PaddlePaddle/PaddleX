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

import copy
import os.path as osp
import numpy as np
import paddle
from paddleslim import L1NormFilterPruner
from . import cv
from .cv.models.utils.visualize import visualize_detection, draw_pr_curve
from paddlex.cv.transforms import det_transforms
from paddlex.cv.transforms.operators import _NormalizeBox, _PadBox, _BboxXYXY2XYWH
from paddlex.cv.transforms.batch_operators import BatchCompose, BatchRandomResize, BatchRandomResizeByShort, \
    _BatchPadding, _Gt2YoloTarget
import paddlex.utils.logging as logging
from paddlex.utils.checkpoint import det_pretrain_weights_dict
from paddlex.cv.models.utils.ema import ExponentialMovingAverage

transforms = det_transforms

visualize = visualize_detection
draw_pr_curve = draw_pr_curve


class FasterRCNN(cv.models.FasterRCNN):
    def __init__(self,
                 num_classes=81,
                 backbone='ResNet50',
                 with_fpn=True,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[32, 64, 128, 256, 512],
                 with_dcn=None,
                 rpn_cls_loss=None,
                 rpn_focal_loss_alpha=None,
                 rpn_focal_loss_gamma=None,
                 rcnn_bbox_loss=None,
                 rcnn_nms=None,
                 keep_top_k=100,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 softnms_sigma=None,
                 bbox_assigner=None,
                 fpn_num_channels=256,
                 input_channel=None,
                 rpn_batch_size_per_im=256,
                 rpn_fg_fraction=0.5,
                 test_pre_nms_top_n=None,
                 test_post_nms_top_n=1000):
        if with_dcn is not None:
            logging.warning(
                "`with_dcn` is deprecated in PaddleX 2.0 and won't take effect. Defaults to False."
            )
        if rpn_cls_loss is not None:
            logging.warning(
                "`rpn_cls_loss` is deprecated in PaddleX 2.0 and won't take effect. "
                "Defaults to 'SigmoidCrossEntropy'.")
        if rpn_focal_loss_alpha is not None or rpn_focal_loss_gamma is not None:
            logging.warning(
                "Focal loss is deprecated in PaddleX 2.0."
                " `rpn_focal_loss_alpha` and `rpn_focal_loss_gamma` won't take effect."
            )
        if rcnn_bbox_loss is not None:
            logging.warning(
                "`rcnn_bbox_loss` is deprecated in PaddleX 2.0 and won't take effect. "
                "Defaults to 'SmoothL1Loss'")
        if rcnn_nms is not None:
            logging.warning(
                "MultiClassSoftNMS is deprecated in PaddleX 2.0. "
                "`rcnn_nms` and `softnms_sigma` won't take effect. MultiClassNMS will be used by default"
            )
        if bbox_assigner is not None:
            logging.warning(
                "`bbox_assigner` is deprecated in PaddleX 2.0 and won't take effect. "
                "Defaults to 'BBoxAssigner'")
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        if isinstance(anchor_sizes[0], int):
            anchor_sizes = [[size] for size in anchor_sizes]
        super(FasterRCNN, self).__init__(
            num_classes=num_classes - 1,
            backbone=backbone,
            with_fpn=with_fpn,
            aspect_ratios=aspect_ratios,
            anchor_sizes=anchor_sizes,
            keep_top_k=keep_top_k,
            nms_threshold=nms_threshold,
            score_threshold=score_threshold,
            fpn_num_channels=fpn_num_channels,
            rpn_batch_size_per_im=rpn_batch_size_per_im,
            rpn_fg_fraction=rpn_fg_fraction,
            test_pre_nms_top_n=test_pre_nms_top_n,
            test_post_nms_top_n=test_post_nms_top_n)

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
              learning_rate=0.0025,
              warmup_steps=500,
              warmup_start_lr=1.0 / 1200,
              lr_decay_epochs=[8, 11],
              lr_decay_gamma=0.1,
              metric=None,
              use_vdl=False,
              early_stop=False,
              early_stop_patience=5,
              sensitivities_file=None,
              pruned_flops=.2):
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
            metric=metric,
            use_vdl=use_vdl,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops)


class YOLOv3(cv.models.YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='MobileNetV1',
                 anchors=None,
                 anchor_masks=None,
                 ignore_threshold=0.7,
                 nms_score_threshold=0.01,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 label_smooth=False,
                 train_random_shapes=[
                     320, 352, 384, 416, 448, 480, 512, 544, 576, 608
                 ],
                 input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        if anchors is None:
            anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                       [59, 119], [116, 90], [156, 198], [373, 326]]
        if anchor_masks is None:
            anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        super(YOLOv3, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            anchors=anchors,
            anchor_masks=anchor_masks,
            ignore_threshold=ignore_threshold,
            nms_score_threshold=nms_score_threshold,
            nms_topk=nms_topk,
            nms_keep_topk=nms_keep_topk,
            nms_iou_threshold=nms_iou_threshold,
            label_smooth=label_smooth)
        self.train_random_shapes = train_random_shapes

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=8,
              eval_dataset=None,
              save_interval_epochs=20,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=1.0 / 8000,
              warmup_steps=1000,
              warmup_start_lr=0.0,
              lr_decay_epochs=[213, 240],
              lr_decay_gamma=0.1,
              metric=None,
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
            metric=metric,
            use_vdl=use_vdl,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops)

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPadding(pad_to_stride=-1), _NormalizeBox(),
                _PadBox(getattr(self, 'num_max_boxes', 50)), _BboxXYXY2XYWH(),
                _Gt2YoloTarget(
                    anchor_masks=self.anchor_masks,
                    anchors=self.anchors,
                    downsample_ratios=getattr(self, 'downsample_ratios',
                                              [32, 16, 8]),
                    num_classes=self.num_classes)
            ]
        else:
            default_batch_transforms = [_BatchPadding(pad_to_stride=-1)]
        if mode == 'eval' and self.metric == 'voc':
            collate_batch = False
        else:
            collate_batch = True

        custom_batch_transforms = []
        random_shape_defined = False
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                if mode != 'train':
                    raise Exception(
                        "{} cannot be present in the {} transforms. ".format(
                            op.__class__.__name__, mode) +
                        "Please check the {} transforms.".format(mode))
                custom_batch_transforms.insert(0, copy.deepcopy(op))
                random_shape_defined = True
        if not random_shape_defined:
            default_batch_transforms.insert(
                0,
                BatchRandomResize(
                    target_sizes=self.train_random_shapes, interp='RANDOM'))

        batch_transforms = BatchCompose(
            custom_batch_transforms + default_batch_transforms,
            collate_batch=collate_batch)

        return batch_transforms


class PPYOLO(cv.models.PPYOLO):
    def __init__(
            self,
            num_classes=80,
            backbone='ResNet50_vd_ssld',
            with_dcn_v2=None,
            # YOLO Head
            anchors=None,
            anchor_masks=None,
            use_coord_conv=True,
            use_iou_aware=True,
            use_spp=True,
            use_drop_block=True,
            scale_x_y=1.05,
            # PPYOLO Loss
            ignore_threshold=0.7,
            label_smooth=False,
            use_iou_loss=True,
            # NMS
            use_matrix_nms=True,
            nms_score_threshold=0.01,
            nms_topk=1000,
            nms_keep_topk=100,
            nms_iou_threshold=0.45,
            train_random_shapes=[
                320, 352, 384, 416, 448, 480, 512, 544, 576, 608
            ],
            input_channel=None):
        if backbone == 'ResNet50_vd_ssld':
            backbone = 'ResNet50_vd_dcn'
        if with_dcn_v2 is not None:
            logging.warning(
                "`with_dcn_v2` is deprecated in PaddleX 2.0 and will not take effect. "
                "To use backbone with deformable convolutional networks, "
                "please specify in `backbone_name`. "
                "Currently the only backbone with dcn is 'ResNet50_vd_dcn'.")
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(PPYOLO, self).__init__(
            num_classes=num_classes,
            backbone=backbone,
            anchors=anchors,
            anchor_masks=anchor_masks,
            use_coord_conv=use_coord_conv,
            use_iou_aware=use_iou_aware,
            use_spp=use_spp,
            use_drop_block=use_drop_block,
            scale_x_y=scale_x_y,
            ignore_threshold=ignore_threshold,
            label_smooth=label_smooth,
            use_iou_loss=use_iou_loss,
            use_matrix_nms=use_matrix_nms,
            nms_score_threshold=nms_score_threshold,
            nms_topk=nms_topk,
            nms_keep_topk=nms_keep_topk,
            nms_iou_threshold=nms_iou_threshold)
        self.train_random_shapes = train_random_shapes

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=8,
              eval_dataset=None,
              save_interval_epochs=20,
              log_interval_steps=2,
              save_dir='output',
              pretrain_weights='IMAGENET',
              optimizer=None,
              learning_rate=1.0 / 8000,
              warmup_steps=1000,
              warmup_start_lr=0.0,
              lr_decay_epochs=[213, 240],
              lr_decay_gamma=0.1,
              metric=None,
              use_vdl=False,
              sensitivities_file=None,
              pruned_flops=.2,
              early_stop=False,
              early_stop_patience=5,
              resume_checkpoint=None,
              use_ema=True,
              ema_decay=0.9998):
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
            metric=metric,
            use_vdl=use_vdl,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience,
            sensitivities_file=sensitivities_file,
            pruned_flops=pruned_flops,
            use_ema=use_ema,
            ema_decay=ema_decay)

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPadding(pad_to_stride=-1), _NormalizeBox(),
                _PadBox(getattr(self, 'num_max_boxes', 50)), _BboxXYXY2XYWH(),
                _Gt2YoloTarget(
                    anchor_masks=self.anchor_masks,
                    anchors=self.anchors,
                    downsample_ratios=getattr(self, 'downsample_ratios',
                                              [32, 16, 8]),
                    num_classes=self.num_classes)
            ]
        else:
            default_batch_transforms = [_BatchPadding(pad_to_stride=-1)]
        if mode == 'eval' and self.metric == 'voc':
            collate_batch = False
        else:
            collate_batch = True

        custom_batch_transforms = []
        random_shape_defined = False
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                if mode != 'train':
                    raise Exception(
                        "{} cannot be present in the {} transforms. ".format(
                            op.__class__.__name__, mode) +
                        "Please check the {} transforms.".format(mode))
                custom_batch_transforms.insert(0, copy.deepcopy(op))
                random_shape_defined = True
        if not random_shape_defined:
            default_batch_transforms.insert(
                0,
                BatchRandomResize(
                    target_sizes=self.train_random_shapes, interp='RANDOM'))

        batch_transforms = BatchCompose(
            custom_batch_transforms + default_batch_transforms,
            collate_batch=collate_batch)

        return batch_transforms


def _legacy_train(model,
                  num_epochs,
                  train_dataset,
                  train_batch_size,
                  eval_dataset,
                  save_interval_epochs,
                  log_interval_steps,
                  save_dir,
                  pretrain_weights,
                  optimizer,
                  learning_rate,
                  warmup_steps,
                  warmup_start_lr,
                  lr_decay_epochs,
                  lr_decay_gamma,
                  metric,
                  use_vdl,
                  early_stop,
                  early_stop_patience,
                  sensitivities_file,
                  pruned_flops,
                  use_ema=False,
                  ema_decay=0.9998):
    if train_dataset.__class__.__name__ == 'VOCDetection':
        train_dataset.data_fields = {
            'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class', 'difficult'
        }
    elif train_dataset.__class__.__name__ == 'CocoDetection':
        if model.__class__.__name__ == 'MaskRCNN':
            train_dataset.data_fields = {
                'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                'gt_poly', 'is_crowd'
            }
        else:
            train_dataset.data_fields = {
                'im_id', 'image_shape', 'image', 'gt_bbox', 'gt_class',
                'is_crowd'
            }

    if metric is None:
        if eval_dataset.__class__.__name__ == 'VOCDetection':
            model.metric = 'voc'
        elif eval_dataset.__class__.__name__ == 'CocoDetection':
            model.metric = 'coco'
    else:
        assert metric.lower() in ['coco', 'voc'], \
            "Evaluation metric {} is not supported, please choose form 'COCO' and 'VOC'"
        model.metric = metric.lower()

    model.labels = train_dataset.labels
    model.num_max_boxes = train_dataset.num_max_boxes
    train_dataset.batch_transforms = model._compose_batch_transform(
        train_dataset.transforms, mode='train')

    if sensitivities_file is not None:
        dataset = eval_dataset or train_dataset
        im_shape = dataset[0]['image'].shape[:2]
        if getattr(model, 'with_fpn', False):
            im_shape[0] = int(np.ceil(im_shape[0] / 32) * 32)
            im_shape[1] = int(np.ceil(im_shape[1] / 32) * 32)
        inputs = [{
            "image": paddle.ones(
                shape=[1, 3] + list(im_shape), dtype='float32'),
            "im_shape": paddle.full(
                [1, 2], 640, dtype='float32'),
            "scale_factor": paddle.ones(
                shape=[1, 2], dtype='float32')
        }]
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

    # initiate weights
    if pretrain_weights is not None and not osp.exists(pretrain_weights):
        if pretrain_weights not in det_pretrain_weights_dict['_'.join(
            [model.model_name, model.backbone_name])]:
            logging.warning("Path of pretrain_weights('{}') does not exist!".
                            format(pretrain_weights))
            pretrain_weights = det_pretrain_weights_dict['_'.join(
                [model.model_name, model.backbone_name])][0]
            logging.warning("Pretrain_weights is forcibly set to '{}'. "
                            "If you don't want to use pretrain weights, "
                            "set pretrain_weights to be None.".format(
                                pretrain_weights))
    pretrained_dir = osp.join(save_dir, 'pretrain')
    model.net_initialize(
        pretrain_weights=pretrain_weights, save_dir=pretrained_dir)

    if use_ema:
        ema = ExponentialMovingAverage(
            decay=ema_decay, model=model.net, use_thres_step=True)
    else:
        ema = None
    # start train loop
    model.train_loop(
        num_epochs=num_epochs,
        train_dataset=train_dataset,
        train_batch_size=train_batch_size,
        eval_dataset=eval_dataset,
        save_interval_epochs=save_interval_epochs,
        log_interval_steps=log_interval_steps,
        save_dir=save_dir,
        ema=ema,
        early_stop=early_stop,
        early_stop_patience=early_stop_patience,
        use_vdl=use_vdl)
