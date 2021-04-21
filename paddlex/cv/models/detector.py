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
import copy
import os.path as osp
from paddle.io import DistributedBatchSampler
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.nets.ppdet.modeling.proposal_generator.target_layer import BBoxAssigner
from paddlex.cv.nets.ppdet.modeling import *
from paddlex.cv.nets.ppdet.modeling.post_process import *
from paddlex.cv.nets.ppdet.modeling.layers import YOLOBox, MultiClassNMS, RCNNBox
from paddlex.utils import get_single_card_bs, _get_shared_memory_size_in_M
from paddlex.cv.transforms.operators import _NormalizeBox, _PadBox, _BboxXYXY2XYWH
from paddlex.cv.transforms.batch_operators import BatchCompose, BatchRandomResize, BatchPadding, _Gt2YoloTarget, _Permute
from paddlex.cv.transforms import arrange_transforms
from .base import BaseModel
from .utils.det_dataloader import BaseDataLoader
from .utils.det_metrics import VOCMetric

__all__ = ["YOLOv3", "FasterRCNN"]


class BaseDetector(BaseModel):
    def __init__(self, model_name, num_classes=80, **params):
        self.init_params.update(locals())
        del self.init_params['params']
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
        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            net = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        test_inputs = [
            paddle.static.InputSpec(
                shape=[None, 3, None, None], dtype='float32')
        ]
        return net, test_inputs

    def _get_backbone(self, backbone_name, **params):
        backbone = backbones.__dict__[backbone_name](**params)
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
        net_out = net(inputs)
        if mode in ['train', 'eval']:
            outputs = net_out
        else:
            for key in ['im_shape', 'scale_factor']:
                net_out[key] = inputs[key]
            outputs = dict()
            for key in net_out:
                outputs[key] = net_out[key].numpy()

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
              pretrain_weights='COCO',
              learning_rate=.001,
              warmup_steps=0,
              warmup_start_lr=0.0,
              lr_decay_epochs=(216, 243),
              lr_decay_gamma=0.1,
              early_stop=False,
              early_stop_patience=5):
        train_dataset.batch_transforms = self._compose_batch_transform(
            train_dataset.transforms, mode='train')
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
            if pretrain_weights not in ['COCO', 'PascalVOC']:
                logging.warning(
                    "Path of pretrain_weights('{}') does not exist!".format(
                        pretrain_weights))
                logging.warning("Pretrain_weights is forcibly set to 'COCO'. "
                                "If don't want to use pretrain weights, "
                                "set pretrain_weights to be None.")
                pretrain_weights = 'COCO'
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
        eval_dataset.batch_transforms = self._compose_batch_transform(
            eval_dataset.transforms, mode='eval')
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
                "during evaluation, so batch_size is forcibly set to 1.")
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

    def predict(self, img_file, transforms=None):
        if transforms is None and not hasattr(self, 'test_transforms'):
            raise Exception("transforms need to be defined, now is None.")
        if transforms is None:
            transforms = self.test_transforms
        if isinstance(img_file, (str, np.ndarray)):
            images = [img_file]
        else:
            images = img_file

        batch_samples = self._preprocess(images, transforms)
        self.net.eval()
        outputs = self.run(self.net, batch_samples, 'test')
        pred = self._postprocess(outputs)

        return pred

    def _preprocess(self, images, transforms):
        arrange_transforms(
            model_type=self.model_type, transforms=transforms, mode='test')
        batch_samples = list()
        for ct, im in enumerate(images):
            sample = {'image': im}
            batch_samples.append(transforms(sample))
        batch_transforms = self._compose_batch_transform(transforms, 'test')
        batch_samples = batch_transforms(batch_samples)
        batch_samples = list(map(paddle.to_tensor, batch_samples))
        batch_samples = {
            k: v
            for k, v in zip(batch_transforms.output_fields, batch_samples)
        }
        return batch_samples

    def _postprocess(self, batch_pred):
        if 'bbox' in batch_pred:
            bboxes = batch_pred['bbox']
            bbox_nums = batch_pred['bbox_num']
            det_res = []
            k = 0
            for i in range(len(bbox_nums)):
                det_nums = bbox_nums[i]
                for j in range(det_nums):
                    dt = bboxes[k]
                    k = k + 1
                    num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                    if int(num_id) < 0:
                        continue
                    category_id = int(num_id)
                    w = xmax - xmin
                    h = ymax - ymin
                    bbox = [xmin, ymin, w, h]
                    dt_res = {
                        'category_id': category_id,
                        'bbox': bbox,
                        'score': score
                    }
                    det_res.append(dt_res)
            batch_pred['bbox'] = det_res

        result = []
        bbox_num = batch_pred['bbox_num']
        start = 0
        for num in bbox_num:
            end = start + num
            if 'bbox' in batch_pred:
                bbox_res = batch_pred['bbox'][start:end]
            result.append(bbox_res)
            start = end

        return result


class YOLOv3(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='MobileNetV1',
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 ignore_threshold=0.7,
                 nms_score_threshold=0.01,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45,
                 label_smooth=False):
        self.init_params = locals()
        if backbone not in [
                'MobileNetV1', 'MobileNetV3', 'DarkNet53', 'ResNet50_vd_dcn'
        ]:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('MobileNetV1', 'MobileNetV3', 'DarkNet53', 'ResNet50_vd_dcn')".
                format(backbone))

        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            norm_type = 'sync_bn'
        else:
            norm_type = 'bn'

        self.backbone_name = backbone
        if backbone == 'MobileNetV1':
            norm_type = 'bn'
            backbone = self._get_backbone('MobileNet', norm_type=norm_type)
        elif backbone == 'MobileNetV3':
            backbone = self._get_backbone(
                'MobileNetV3', norm_type=norm_type, feature_maps=[7, 13, 16])
        elif backbone == 'ResNet50_vd_dcn':
            backbone = self._get_backbone(
                'ResNet',
                norm_type=norm_type,
                variant='d',
                return_idx=[1, 2, 3],
                dcn_v2_stages=[3],
                freeze_at=-1,
                freeze_norm=False)
        else:
            backbone = self._get_backbone('DarkNet', norm_type=norm_type)

        neck = necks.YOLOv3FPN(
            norm_type=norm_type,
            in_channels=[i.channels for i in backbone.out_shape])
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

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _NormalizeBox(), _PadBox(50), _BboxXYXY2XYWH(), _Gt2YoloTarget(
                    anchor_masks=self.anchors_masks,
                    anchors=self.anchors,
                    downsample_ratios=[32, 16, 8],
                    num_classes=6)
            ]
        else:
            default_batch_transforms = []

        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, BatchRandomResize):
                custom_batch_transforms.insert(0, copy.deepcopy(op))
            elif isinstance(op, BatchPadding):
                custom_batch_transforms.insert(-1, copy.deepcopy(op))
        batch_transforms = BatchCompose(custom_batch_transforms +
                                        default_batch_transforms)

        return batch_transforms


class FasterRCNN(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50',
                 with_fpn=True,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[32, 64, 128, 256, 512],
                 keep_top_k=100,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 fpn_num_channels=256,
                 rpn_batch_size_per_im=256,
                 rpn_fg_fraction=0.5,
                 test_pre_nms_top_n=None,
                 test_post_nms_top_n=1000):
        self.init_params = locals()
        if backbone not in ['ResNet50']:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('ResNet50')".format(backbone))

        self.backbone_name = backbone
        if backbone == 'ResNet50':
            if with_fpn:
                backbone = self._get_backbone(
                    'ResNet',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4)
            else:
                backbone = self._get_backbone(
                    'ResNet',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[2],
                    num_stages=3)
        else:
            backbone = self._get_backbone(backbone)

        rpn_in_channel = backbone.out_shape[0].channels

        if with_fpn:
            neck = necks.FPN(
                in_channels=[i.channels for i in backbone.out_shape],
                out_channel=fpn_num_channels,
                spatial_scales=[1.0 / i.stride for i in backbone.out_shape])
            rpn_in_channel = neck.out_shape[0].channels
            anchor_generator_cfg = {
                'aspect_ratios': aspect_ratios,
                'anchor_sizes': anchor_sizes,
                'strides': [4, 8, 16, 32, 64]
            }
            train_proposal_cfg = {
                'min_size': 0.0,
                'nms_thresh': .7,
                'pre_nms_top_n': 2000,
                'post_nms_top_n': 1000,
                'topk_after_collect': True
            }
            test_proposal_cfg = {
                'min_size': 0.0,
                'nms_thresh': .7,
                'pre_nms_top_n': 1000
                if test_pre_nms_top_n is None else test_pre_nms_top_n,
                'post_nms_top_n': test_post_nms_top_n
            }
            head = heads.TwoFCHead(out_channel=1024)
            roi_extractor_cfg = {
                'resolution': 7,
                'spatial_scale': [1. / i.stride for i in backbone.out_shape],
                'sampling_ratio': 0,
                'aligned': True
            }
            with_pool = False

        else:
            neck = None
            anchor_generator_cfg = {
                'aspect_ratios': aspect_ratios,
                'anchor_sizes': anchor_sizes,
                'strides': [16]
            }
            train_proposal_cfg = {
                'min_size': 0.0,
                'nms_thresh': .7,
                'pre_nms_top_n': 12000,
                'post_nms_top_n': 2000,
                'topk_after_collect': False
            }
            test_proposal_cfg = {
                'min_size': 0.0,
                'nms_thresh': .7,
                'pre_nms_top_n': 6000
                if test_pre_nms_top_n is None else test_pre_nms_top_n,
                'post_nms_top_n': test_post_nms_top_n
            }
            head = backbones.Res5Head()
            roi_extractor_cfg = {
                'resolution': 14,
                'spatial_scale': [1. / i.stride for i in backbone.out_shape],
                'sampling_ratio': 0,
                'aligned': True
            }
            with_pool = True

        rpn_target_assign_cfg = {
            'batch_size_per_im': rpn_batch_size_per_im,
            'fg_fraction': rpn_fg_fraction,
            'negative_overlap': .3,
            'positive_overlap': .7,
            'use_random': True
        }

        rpn_head = RPNHead(
            anchor_generator=anchor_generator_cfg,
            rpn_target_assign=rpn_target_assign_cfg,
            train_proposal=train_proposal_cfg,
            test_proposal=test_proposal_cfg,
            in_channel=rpn_in_channel)

        bbox_assigner = BBoxAssigner(num_classes=num_classes)

        bbox_head = heads.BBoxHead(
            head=head,
            in_channel=head.out_shape[0].channels,
            roi_extractor=roi_extractor_cfg,
            with_pool=with_pool,
            bbox_assigner=bbox_assigner)

        bbox_post_process = BBoxPostProcess(
            num_classes=num_classes,
            decode=RCNNBox(num_classes=num_classes),
            nms=MultiClassNMS(
                score_threshold=score_threshold,
                keep_top_k=keep_top_k,
                nms_threshold=nms_threshold))

        params = {
            'backbone': backbone,
            'neck': neck,
            'rpn_head': rpn_head,
            'bbox_head': bbox_head,
            'bbox_post_process': bbox_post_process
        }
        super(FasterRCNN, self).__init__(
            model_name='FasterRCNN', num_classes=num_classes, **params)

    def _compose_batch_transform(self, transforms, mode='train'):
        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, BatchRandomResize):
                custom_batch_transforms.insert(0, copy.deepcopy(op))
            elif isinstance(op, BatchPadding):
                custom_batch_transforms.insert(-1, copy.deepcopy(op))

        batch_transforms = BatchCompose(custom_batch_transforms)

        return batch_transforms
