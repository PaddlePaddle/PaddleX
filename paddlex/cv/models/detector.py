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

from __future__ import absolute_import

import collections
import os.path as osp
import math
import paddle
from paddle.io import DistributedBatchSampler
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.nets.ppdet.modeling import architectures, backbones, necks, heads, losses
from paddlex.cv.nets.ppdet.modeling.post_process import *
from paddlex.cv.nets.ppdet.modeling.layers import YOLOBox, MultiClassNMS
from paddlex.utils import get_single_card_bs, _get_shared_memory_size_in_M
from paddlex.cv.transforms.operators import _NormalizeBox, _PadBox, _BboxXYXY2XYWH
from paddlex.cv.transforms.batch_operators import BatchCompose, BatchRandomResize, _Gt2YoloTarget, _Permute
from paddlex.cv.transforms import arrange_transforms
from .base import BaseModel
from .utils.det_dataloader import BaseDataLoader
from .utils.det_metrics import VOCMetric


class BaseDetector(BaseModel):
    def __init__(self, model_name, num_classes=80, **params):
        self.init_params = locals()
        super(BaseDetector, self).__init__('detector')
        if not hasattr(architectures, model_name):
            raise Exception("ERROR: There's no model named {}.".format(
                model_name))

        self.model_name = model_name
        self.num_classes = num_classes
        self.labels = None
        self.net, self.test_inputs = self.build_net(**params)

    def build_net(self, **params):
        net = architectures.__dict__[self.model_name](**params)
        if self.sync_bn:
            net = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        test_inputs = [
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype='float32')
        ]
        return net, test_inputs

    def _get_backbone(self, backbone_name, norm_type):
        if backbone_name == 'MobileNetV1':
            backbone = backbones.MobileNet(norm_type=norm_type)

        return backbone

    def build_data_loader(self, dataset, batch_size, mode='train'):
        batch_size_each_card = get_single_card_bs(batch_size=batch_size)
        if mode == 'eval':
            batch_size = batch_size_each_card * paddlex.env_info['num']
            total_steps = math.ceil(dataset.num_samples * 1.0 / batch_size)
            logging.info(
                "Start to evaluating(total_samples={}, total_steps={})...".
                format(dataset.num_samples, total_steps))
        if dataset.num_samples < batch_size:
            raise Exception(
                'The volume of datset({}) must be larger than batch size({}).'
                .format(dataset.num_samples, batch_size))

        # TODO detection eval阶段需做判断
        batch_sampler = DistributedBatchSampler(
            dataset,
            batch_size=batch_size_each_card,
            shuffle=dataset.shuffle,
            drop_last=mode == 'train')

        shm_size = _get_shared_memory_size_in_M()
        if shm_size is None or shm_size < 1024.:
            use_shared_memory = False
        else:
            use_shared_memory = True

        loader = BaseDataLoader(
            dataset,
            batch_sampler=batch_sampler,
            use_shared_memory=use_shared_memory)

        return loader

    def run(self, net, inputs, mode):
        if mode == 'train':
            outputs = net(inputs)
        elif mode == 'eval':
            net_out = net(inputs)
        else:
            pass

        return outputs

    def default_optimizer(self, parameters, learning_rate, warmup_steps,
                          warmup_start_lr, lr_decay_epochs, lr_decay_gamma,
                          num_steps_each_epoch):
        boundaries = [b * num_steps_each_epoch for b in lr_decay_epochs]
        values = [(lr_decay_gamma**i) * learning_rate
                  for i in range(len(lr_decay_epochs) + 1)]
        scheduler = paddle.optimizer.lr.PiecewiseDecay(
            boundaries=boundaries, values=values)
        if warmup_steps > 0:
            if warmup_steps > lr_decay_epochs[0] * num_steps_each_epoch:
                logging.error(
                    "In function train(), parameters should satisfy: "
                    "warmup_steps <= lr_decay_epochs[0]*num_samples_in_train_dataset",
                    exit=False)
                logging.error(
                    "See this doc for more information: "
                    "https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/appendix/parameters.md#notice",
                    exit=False)
                logging.error(
                    "warmup_steps should less than {} or lr_decay_epochs[0] greater than {}, "
                    "please modify 'lr_decay_epochs' or 'warmup_steps' in train function".
                    format(lr_decay_epochs[0] * num_steps_each_epoch,
                           warmup_steps // num_steps_each_epoch))
            scheduler = paddle.optimizer.lr.LinearWarmup(
                learning_rate=scheduler,
                warmup_steps=warmup_steps,
                start_lr=warmup_start_lr,
                end_lr=learning_rate)
        optimizer = paddle.optimizer.Momentum(
            scheduler,
            momentum=.9,
            weight_decay=paddle.regularizer.L2Decay(coeff=1e-04),
            parameters=parameters)
        return optimizer

    def train(self,
              num_epochs,
              train_dataset,
              train_batch_size=64,
              eval_dataset=None,
              optimizer=None,
              save_interval_epochs=1,
              log_interval_steps=10,
              save_dir='output',
              pretrain_weights='IMAGENET',
              learning_rate=.001,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=(216, 243),
              lr_decay_gamma=0.1,
              early_stop=False,
              early_stop_patience=5):
        self._arrange_batch_transform(train_dataset, mode='train')
        self.labels = train_dataset.labels

        # build optimizer if not defined
        if optimizer is None:
            num_steps_each_epoch = len(train_dataset) // train_batch_size
            self.optimizer = self.default_optimizer(
                parameters=self.net.parameters(),
                learning_rate=learning_rate,
                warmup_steps=warmup_steps,
                warmup_start_lr=warmup_start_lr,
                lr_decay_epochs=lr_decay_epochs,
                lr_decay_gamma=lr_decay_gamma,
                num_steps_each_epoch=num_steps_each_epoch)
        else:
            self.optimizer = optimizer

        # initiate weights
        if pretrain_weights is not None and not osp.exists(pretrain_weights):
            if pretrain_weights not in ['IMAGENET']:
                logging.warning(
                    "Path of pretrain_weights('{}') does not exist!".format(
                        pretrain_weights))
                logging.warning(
                    "Pretrain_weights is forcibly set to 'IMAGENET'. "
                    "If don't want to use pretrain weights, "
                    "set pretrain_weights to be None.")
                pretrain_weights = 'IMAGENET'
        pretrained_dir = osp.join(save_dir, 'pretrain')
        self.net_initialize(
            pretrain_weights=pretrain_weights, save_dir=pretrained_dir)

        # start train loop
        self.train_loop(
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            train_batch_size=train_batch_size,
            eval_dataset=eval_dataset,
            save_interval_epochs=save_interval_epochs,
            log_interval_steps=log_interval_steps,
            save_dir=save_dir,
            early_stop=early_stop,
            early_stop_patience=early_stop_patience)

    def evaluate(self, eval_dataset, batch_size, return_details=False):
        self._arrange_batch_transform(eval_dataset, mode='eval')
        arrange_transforms(
            model_type=self.model_type,
            transforms=eval_dataset.transforms,
            mode='eval')

        self.net.eval()
        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank
        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()

        if batch_size > 1:
            logging.warning(
                "Detector only supports single card evaluation with batch_size=1 "
                "during evaluation, so batch_size is forcibly set to 1")
            batch_size = 1

        if nranks < 2 or local_rank == 0:
            self.eval_data_loader = self.build_data_loader(
                eval_dataset, batch_size=batch_size, mode='eval')
            is_bbox_normalized = False
            if eval_dataset.batch_transforms is not None:
                is_bbox_normalized = any(
                    isinstance(t, _NormalizeBox)
                    for t in eval_dataset.batch_transforms.batch_transforms)
            eval_metrics = [
                VOCMetric(
                    labels=eval_dataset.labels,
                    is_bbox_normalized=is_bbox_normalized,
                    classwise=False)
            ]
            scores = collections.OrderedDict()
            with paddle.no_grad():
                for step, data in enumerate(self.eval_data_loader):
                    outputs = self.run(self.net, data, 'eval')
                    for metric in eval_metrics:
                        metric.update(data, outputs)
                for metric in eval_metrics:
                    metric.accumulate()
                    scores.update(metric.get())
                    metric.reset()
            return scores


class YOLOv3(BaseDetector):
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
                 label_smooth=False):
        if backbone not in ['MobileNetV1']:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('MobileNetV1')".format(backbone))

        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            self.sync_bn = True
        else:
            self.sync_bn = False
        backbone = self._get_backbone(backbone)
        neck = necks.YOLOv3FPN()
        loss = losses.YOLOv3Loss(
            num_classes=num_classes,
            ignore_thresh=ignore_threshold,
            label_smooth=label_smooth)
        yolo_head = heads.YOLOv3Head(
            anchors=anchors,
            anchor_masks=anchor_masks,
            num_classes=num_classes,
            loss=loss)
        post_process = BBoxPostProcess(
            decode=YOLOBox(num_classes=num_classes),
            nms=MultiClassNMS(
                score_threshold=nms_score_threshold,
                nms_top_k=nms_topk,
                keep_top_k=nms_keep_topk,
                nms_threshold=nms_iou_threshold))
        params = {
            'backbone': backbone,
            'neck': neck,
            'yolo_head': yolo_head,
            'post_process': post_process
        }
        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchors_masks = anchor_masks

    def _arrange_batch_transform(self, dataset, mode='train'):
        if mode == 'train':
            batch_transforms = [
                _NormalizeBox(), _PadBox(50), _BboxXYXY2XYWH(), _Gt2YoloTarget(
                    anchor_masks=self.anchors_masks,
                    anchors=self.anchors,
                    downsample_ratios=[32, 16, 8],
                    num_classes=6), _Permute()
            ]
        elif mode == 'eval':
            batch_transforms = [_Permute()]

        for i, op in enumerate(dataset.transforms.transforms):
            if isinstance(op, BatchRandomResize):
                batch_transforms.insert(0,
                                        dataset.transforms.transforms.pop(i))
                break

        dataset.batch_transforms = BatchCompose(batch_transforms)
