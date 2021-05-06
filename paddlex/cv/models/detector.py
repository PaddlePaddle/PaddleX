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
import pycocotools.mask as mask_util
from paddle.io import DistributedBatchSampler
import paddlex
import paddlex.utils.logging as logging
from paddlex.cv.nets.ppdet.modeling.proposal_generator.target_layer import BBoxAssigner, MaskAssigner
from paddlex.cv.nets.ppdet.modeling import *
from paddlex.cv.nets.ppdet.modeling.post_process import *
from paddlex.cv.nets.ppdet.modeling.layers import YOLOBox, MultiClassNMS, RCNNBox
from paddlex.utils import get_single_card_bs, _get_shared_memory_size_in_M
from paddlex.cv.transforms.operators import _NormalizeBox, _PadBox, _BboxXYXY2XYWH
from paddlex.cv.transforms.batch_operators import BatchCompose, BatchRandomResize, BatchRandomResizeByShort, _BatchPadding, _Gt2YoloTarget, _Permute
from paddlex.cv.transforms import arrange_transforms
from .base import BaseModel
from .utils.det_dataloader import BaseDataLoader
from .utils.det_metrics import VOCMetric, COCOMetric
from paddlex.utils.checkpoint import det_pretrain_weights_dict

__all__ = [
    "YOLOv3", "FasterRCNN", "PPYOLO", "PPYOLOTiny", "PPYOLOv2", "MaskRCNN"
]


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
              early_stop_patience=5,
              use_vdl=True):
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
            if pretrain_weights not in det_pretrain_weights_dict['_'.join(
                [self.model_name, self.backbone_name])]:
                logging.warning(
                    "Path of pretrain_weights('{}') does not exist!".format(
                        pretrain_weights))
                pretrain_weights = det_pretrain_weights_dict['_'.join(
                    [self.model_name, self.backbone_name])][0]
                logging.warning("Pretrain_weights is forcibly set to '{}'. "
                                "If don't want to use pretrain weights, "
                                "set pretrain_weights to be None.".format(
                                    pretrain_weights))
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
            early_stop_patience=early_stop_patience,
            use_vdl=use_vdl)

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
            if eval_dataset.__class__.__name__ == 'VOCDetection':
                eval_metrics = [
                    VOCMetric(
                        labels=eval_dataset.labels,
                        is_bbox_normalized=is_bbox_normalized,
                        classwise=False)
                ]
            elif eval_dataset.__class__.__name__ == 'CocoDetection':
                eval_metrics = [
                    COCOMetric(
                        coco_gt=eval_dataset.coco_gt, classwise=False)
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
        for im in images:
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
        infer_result = {}
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
            infer_result['bbox'] = det_res

        if 'mask' in batch_pred:
            masks = batch_pred['mask']
            bboxes = batch_pred['bbox']
            mask_nums = batch_pred['bbox_num']
            seg_res = []
            k = 0
            for i in range(len(mask_nums)):
                det_nums = mask_nums[i]
                for j in range(det_nums):
                    mask = masks[k].astype(np.uint8)
                    score = float(bboxes[k][1])
                    label = int(bboxes[k][0])
                    k = k + 1
                    if label == -1:
                        continue
                    category_id = int(label)
                    rle = mask_util.encode(
                        np.array(
                            mask[:, :, None], order="F", dtype="uint8"))[0]
                    if six.PY3:
                        if 'counts' in rle:
                            rle['counts'] = rle['counts'].decode("utf8")
                    sg_res = {
                        'category_id': category_id,
                        'segmentation': rle,
                        'score': score
                    }
                    seg_res.append(sg_res)
            infer_result['mask'] = seg_res

        bbox_num = batch_pred['bbox_num']
        result = []
        start = 0
        for num in bbox_num:
            curr_result = {}
            end = start + num
            if 'bbox' in infer_result:
                bbox_res = infer_result['bbox'][start:end]
                curr_result['bboxes'] = bbox_res
            if 'mask' in infer_result:
                mask_res = infer_result['mask'][start:end]
                curr_result['masks'] = mask_res
            result.append(curr_result)
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
                'MobileNetV1', 'MobileNetV1_ssld', 'MobileNetV3',
                'MobileNetV3_ssld', 'DarkNet53', 'ResNet50_vd_dcn', 'ResNet34'
        ]:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('MobileNetV1', 'MobileNetV1_ssld', 'MobileNetV3', 'MobileNetV3_ssld', 'DarkNet53', 'ResNet50_vd_dcn', 'ResNet34')".
                format(backbone))

        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            norm_type = 'sync_bn'
        else:
            norm_type = 'bn'

        self.backbone_name = backbone
        if 'MobileNetV1' in backbone:
            norm_type = 'bn'
            backbone = self._get_backbone('MobileNet', norm_type=norm_type)
        elif 'MobileNetV3' in backbone:
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
        elif backbone == 'ResNet34':
            backbone = self._get_backbone(
                'ResNet',
                depth=34,
                norm_type=norm_type,
                return_idx=[1, 2, 3],
                freeze_at=-1,
                freeze_norm=False,
                norm_decay=0.)
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
            in_channels=[i.channels for i in neck.out_shape],
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
        self.anchor_masks = anchor_masks

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPadding(
                    pad_to_stride=-1, pad_gt=False), _NormalizeBox(),
                _PadBox(getattr(self, 'num_max_boxes', 50)), _BboxXYXY2XYWH(),
                _Gt2YoloTarget(
                    anchor_masks=self.anchor_masks,
                    anchors=self.anchors,
                    downsample_ratios=getattr(self, 'downsample_ratios',
                                              [32, 16, 8]),
                    num_classes=self.num_classes)
            ]
        else:
            default_batch_transforms = [
                _BatchPadding(
                    pad_to_stride=-1, pad_gt=False)
            ]

        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(custom_batch_transforms +
                                        default_batch_transforms)

        return batch_transforms


class FasterRCNN(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50',
                 with_fpn=True,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[[32], [64], [128], [256], [512]],
                 keep_top_k=100,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 fpn_num_channels=256,
                 rpn_batch_size_per_im=256,
                 rpn_fg_fraction=0.5,
                 test_pre_nms_top_n=None,
                 test_post_nms_top_n=1000):
        self.init_params = locals()
        if backbone not in [
                'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet34',
                'ResNet34_vd', 'ResNet101', 'ResNet101_vd'
        ]:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet34', 'ResNet34_vd', "
                "'ResNet101', 'ResNet101_vd')".format(backbone))
        self.backbone_name = backbone + '_fpn' if with_fpn else backbone
        if backbone == 'ResNet50_vd_ssld':
            if not with_fpn:
                logging.warning(
                    "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                    format(backbone))
                with_fpn = True
            backbone = self._get_backbone(
                'ResNet',
                variant='d',
                norm_type='bn',
                freeze_at=0,
                return_idx=[0, 1, 2, 3],
                num_stages=4,
                lr_mult_list=[0.05, 0.05, 0.1, 0.15])
        elif 'ResNet50' in backbone:
            if with_fpn:
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d' if '_vd' in backbone else 'b',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[0, 1, 2, 3],
                    num_stages=4)
            else:
                backbone = self._get_backbone(
                    'ResNet',
                    variant='d' if '_vd' in backbone else 'b',
                    norm_type='bn',
                    freeze_at=0,
                    return_idx=[2],
                    num_stages=3)
        elif 'ResNet34' in backbone:
            if not with_fpn:
                logging.warning(
                    "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                    format(backbone))
                with_fpn = True
            backbone = self._get_backbone(
                'ResNet',
                depth=34,
                variant='d' if 'vd' in backbone else 'b',
                norm_type='bn',
                freeze_at=0,
                return_idx=[0, 1, 2, 3],
                num_stages=4)
        else:
            if not with_fpn:
                logging.warning(
                    "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                    format(backbone))
                with_fpn = True
            backbone = self._get_backbone(
                'ResNet',
                depth=101,
                variant='d' if 'vd' in backbone else 'b',
                norm_type='bn',
                freeze_at=0,
                return_idx=[0, 1, 2, 3],
                num_stages=4)

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
                'spatial_scale': [1. / i.stride for i in neck.out_shape],
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
            bbox_assigner=bbox_assigner,
            num_classes=num_classes)

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

        self.with_fpn = with_fpn
        super(FasterRCNN, self).__init__(
            model_name='FasterRCNN', num_classes=num_classes, **params)

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPadding(
                    pad_to_stride=32 if self.with_fpn else -1, pad_gt=True)
            ]
        else:
            default_batch_transforms = [
                _BatchPadding(
                    pad_to_stride=32 if self.with_fpn else -1, pad_gt=False)
            ]
        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(custom_batch_transforms +
                                        default_batch_transforms)

        return batch_transforms


class PPYOLO(YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50_vd_dcn',
                 anchors=None,
                 anchor_masks=None,
                 use_coord_conv=True,
                 use_iou_aware=True,
                 use_spp=True,
                 use_drop_block=True,
                 scale_x_y=1.05,
                 ignore_threshold=0.7,
                 label_smooth=False,
                 use_iou_loss=True,
                 use_matrix_nms=True,
                 nms_score_threshold=0.01,
                 nms_topk=-1,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45):
        self.init_params = locals()
        if backbone not in [
                'ResNet50_vd_dcn', 'ResNet18_vd', 'MobileNetV3_large',
                'MobileNetV3_small'
        ]:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('ResNet50_vd_dcn', 'ResNet18_vd', 'MobileNetV3_large', 'MobileNetV3_small')".
                format(backbone))
        self.backbone_name = backbone

        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            norm_type = 'sync_bn'
        else:
            norm_type = 'bn'
        if anchors is None and anchor_masks is None:
            if 'MobileNetV3' in backbone:
                anchors = [[11, 18], [34, 47], [51, 126], [115, 71],
                           [120, 195], [254, 235]]
                anchor_masks = [[3, 4, 5], [0, 1, 2]]
            elif backbone == 'ResNet50_vd_dcn':
                anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                           [59, 119], [116, 90], [156, 198], [373, 326]]
                anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            else:
                anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169],
                           [344, 319]]
                anchor_masks = [[3, 4, 5], [0, 1, 2]]
        elif anchors is None or anchor_masks is None:
            raise ValueError("Please define both anchors and anchor_masks.")

        if backbone == 'ResNet50_vd_dcn':
            backbone = self._get_backbone(
                'ResNet',
                variant='d',
                norm_type=norm_type,
                return_idx=[1, 2, 3],
                dcn_v2_stages=[3],
                freeze_at=-1,
                freeze_norm=False,
                norm_decay=0.)
            downsample_ratios = [32, 16, 8]

        elif backbone == 'ResNet18_vd':
            backbone = self._get_backbone(
                'ResNet',
                depth=18,
                variant='d',
                norm_type=norm_type,
                return_idx=[2, 3],
                freeze_at=-1,
                freeze_norm=False,
                norm_decay=0.)
            downsample_ratios = [32, 16, 8]

        elif backbone == 'MobileNetV3_large':
            backbone = self._get_backbone(
                'MobileNetV3',
                model_name='large',
                norm_type=norm_type,
                scale=1,
                with_extra_blocks=False,
                extra_block_filters=[],
                feature_maps=[13, 16])
            downsample_ratios = [32, 16]

        elif backbone == 'MobileNetV3_small':
            backbone = self._get_backbone(
                'MobileNetV3',
                model_name='small',
                norm_type=norm_type,
                scale=1,
                with_extra_blocks=False,
                extra_block_filters=[],
                feature_maps=[9, 12])
            downsample_ratios = [32, 16]

        neck = necks.PPYOLOFPN(
            norm_type=norm_type,
            in_channels=[i.channels for i in backbone.out_shape],
            coord_conv=use_coord_conv,
            drop_block=use_drop_block,
            spp=use_spp,
            conv_block_num=0 if ('MobileNetV3' in self.backbone_name or
                                 self.backbone_name == 'ResNet18_vd') else 2)

        loss = losses.YOLOv3Loss(
            num_classes=num_classes,
            ignore_thresh=ignore_threshold,
            downsample=downsample_ratios,
            label_smooth=label_smooth,
            scale_x_y=scale_x_y,
            iou_loss=losses.IouLoss(
                loss_weight=2.5, loss_square=True) if use_iou_loss else None,
            iou_aware_loss=losses.IouAwareLoss(loss_weight=1.0)
            if use_iou_aware else None)

        yolo_head = heads.YOLOv3Head(
            in_channels=[i.channels for i in neck.out_shape],
            anchors=anchors,
            anchor_masks=anchor_masks,
            num_classes=num_classes,
            loss=loss,
            iou_aware=use_iou_aware)

        if use_matrix_nms:
            nms = MatrixNMS(
                keep_top_k=nms_keep_topk,
                score_threshold=nms_score_threshold,
                post_threshold=.05
                if 'MobileNetV3' in self.backbone_name else .01,
                nms_top_k=nms_topk,
                background_label=-1)
        else:
            nms = MultiClassNMS(
                score_threshold=nms_score_threshold,
                nms_top_k=nms_topk,
                keep_top_k=nms_keep_topk,
                nms_threshold=nms_iou_threshold)

        post_process = BBoxPostProcess(
            decode=YOLOBox(
                num_classes=num_classes,
                conf_thresh=.005
                if 'MobileNetV3' in self.backbone_name else .01,
                scale_x_y=scale_x_y),
            nms=nms)

        params = {
            'backbone': backbone,
            'neck': neck,
            'yolo_head': yolo_head,
            'post_process': post_process
        }

        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.model_name = 'PPYOLO'


class PPYOLOTiny(YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='MobileNetV3',
                 anchors=[[10, 15], [24, 36], [72, 42], [35, 87], [102, 96],
                          [60, 170], [220, 125], [128, 222], [264, 266]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 use_iou_aware=False,
                 use_spp=True,
                 use_drop_block=True,
                 scale_x_y=1.05,
                 ignore_threshold=0.5,
                 label_smooth=False,
                 use_iou_loss=True,
                 use_matrix_nms=False,
                 nms_score_threshold=0.005,
                 nms_topk=1000,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45):
        self.init_params = locals()
        if backbone != 'MobileNetV3':
            logging.warning(
                "PPYOLOTiny only supports MobileNetV3 as backbone. "
                "Backbone is forcibly set to MobileNetV3.")
        self.backbone_name = 'MobileNetV3'
        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            norm_type = 'sync_bn'
        else:
            norm_type = 'bn'

        backbone = self._get_backbone(
            'MobileNetV3',
            model_name='large',
            norm_type=norm_type,
            scale=.5,
            with_extra_blocks=False,
            extra_block_filters=[],
            feature_maps=[7, 13, 16])
        downsample_ratios = [32, 16, 8]

        neck = necks.PPYOLOTinyFPN(
            detection_block_channels=[160, 128, 96],
            in_channels=[i.channels for i in backbone.out_shape],
            spp=use_spp,
            drop_block=use_drop_block)

        loss = losses.YOLOv3Loss(
            num_classes=num_classes,
            ignore_thresh=ignore_threshold,
            downsample=downsample_ratios,
            label_smooth=label_smooth,
            scale_x_y=scale_x_y,
            iou_loss=losses.IouLoss(
                loss_weight=2.5, loss_square=True) if use_iou_loss else None,
            iou_aware_loss=losses.IouAwareLoss(loss_weight=1.0)
            if use_iou_aware else None)

        yolo_head = heads.YOLOv3Head(
            in_channels=[i.channels for i in neck.out_shape],
            anchors=anchors,
            anchor_masks=anchor_masks,
            num_classes=num_classes,
            loss=loss,
            iou_aware=use_iou_aware)

        if use_matrix_nms:
            nms = MatrixNMS(
                keep_top_k=nms_keep_topk,
                score_threshold=nms_score_threshold,
                post_threshold=.05,
                nms_top_k=nms_topk,
                background_label=-1)
        else:
            nms = MultiClassNMS(
                score_threshold=nms_score_threshold,
                nms_top_k=nms_topk,
                keep_top_k=nms_keep_topk,
                nms_threshold=nms_iou_threshold)

        post_process = BBoxPostProcess(
            decode=YOLOBox(
                num_classes=num_classes,
                conf_thresh=.005,
                downsample_ratio=32,
                clip_bbox=True,
                scale_x_y=scale_x_y),
            nms=nms)

        params = {
            'backbone': backbone,
            'neck': neck,
            'yolo_head': yolo_head,
            'post_process': post_process
        }

        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_max_boxes = 100
        self.model_name = 'PPYOLOTiny'


class PPYOLOv2(YOLOv3):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50_vd_dcn',
                 anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                          [59, 119], [116, 90], [156, 198], [373, 326]],
                 anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
                 use_iou_aware=True,
                 use_spp=True,
                 use_drop_block=True,
                 scale_x_y=1.05,
                 ignore_threshold=0.7,
                 label_smooth=False,
                 use_iou_loss=True,
                 use_matrix_nms=True,
                 nms_score_threshold=0.01,
                 nms_topk=-1,
                 nms_keep_topk=100,
                 nms_iou_threshold=0.45):
        self.init_params = locals()
        if backbone not in ['ResNet50_vd_dcn', 'ResNet101_vd_dcn']:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('ResNet50_vd_dcn', 'ResNet18_vd')".format(backbone))
        self.backbone_name = backbone

        if paddlex.env_info['place'] == 'gpu' and paddlex.env_info['num'] > 1:
            norm_type = 'sync_bn'
        else:
            norm_type = 'bn'

        if backbone == 'ResNet50_vd_dcn':
            backbone = self._get_backbone(
                'ResNet',
                variant='d',
                norm_type=norm_type,
                return_idx=[1, 2, 3],
                dcn_v2_stages=[3],
                freeze_at=-1,
                freeze_norm=False,
                norm_decay=0.)
            downsample_ratios = [32, 16, 8]

        elif backbone == 'ResNet101_vd_dcn':
            backbone = self._get_backbone(
                'ResNet',
                depth=101,
                variant='d',
                norm_type=norm_type,
                return_idx=[1, 2, 3],
                dcn_v2_stages=[3],
                freeze_at=-1,
                freeze_norm=False,
                norm_decay=0.)
            downsample_ratios = [32, 16, 8]

        neck = necks.PPYOLOPAN(
            norm_type=norm_type,
            in_channels=[i.channels for i in backbone.out_shape],
            drop_block=use_drop_block,
            block_size=3,
            keep_prob=.9,
            spp=use_spp)

        loss = losses.YOLOv3Loss(
            num_classes=num_classes,
            ignore_thresh=ignore_threshold,
            downsample=downsample_ratios,
            label_smooth=label_smooth,
            scale_x_y=scale_x_y,
            iou_loss=losses.IouLoss(
                loss_weight=2.5, loss_square=True) if use_iou_loss else None,
            iou_aware_loss=losses.IouAwareLoss(loss_weight=1.0)
            if use_iou_aware else None)

        yolo_head = heads.YOLOv3Head(
            in_channels=[i.channels for i in neck.out_shape],
            anchors=anchors,
            anchor_masks=anchor_masks,
            num_classes=num_classes,
            loss=loss,
            iou_aware=use_iou_aware,
            iou_aware_factor=.5)

        if use_matrix_nms:
            nms = MatrixNMS(
                keep_top_k=nms_keep_topk,
                score_threshold=nms_score_threshold,
                post_threshold=.01,
                nms_top_k=nms_topk,
                background_label=-1)
        else:
            nms = MultiClassNMS(
                score_threshold=nms_score_threshold,
                nms_top_k=nms_topk,
                keep_top_k=nms_keep_topk,
                nms_threshold=nms_iou_threshold)

        post_process = BBoxPostProcess(
            decode=YOLOBox(
                num_classes=num_classes,
                conf_thresh=.01,
                downsample_ratio=32,
                clip_bbox=True,
                scale_x_y=scale_x_y),
            nms=nms)

        params = {
            'backbone': backbone,
            'neck': neck,
            'yolo_head': yolo_head,
            'post_process': post_process
        }

        super(YOLOv3, self).__init__(
            model_name='YOLOv3', num_classes=num_classes, **params)
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_max_boxes = 100
        self.model_name = 'PPYOLOv2'


class MaskRCNN(BaseDetector):
    def __init__(self,
                 num_classes=80,
                 backbone='ResNet50_vd',
                 with_fpn=True,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 anchor_sizes=[[32], [64], [128], [256], [512]],
                 keep_top_k=100,
                 nms_threshold=0.5,
                 score_threshold=0.05,
                 fpn_num_channels=256,
                 rpn_batch_size_per_im=256,
                 rpn_fg_fraction=0.5,
                 test_pre_nms_top_n=None,
                 test_post_nms_top_n=1000):
        self.init_params = locals()
        if backbone not in [
                'ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101',
                'ResNet101_vd'
        ]:
            raise ValueError(
                "backbone: {} is not supported. Please choose one of "
                "('ResNet50', 'ResNet50_vd', 'ResNet50_vd_ssld', 'ResNet101', 'ResNet101_vd')".
                format(backbone))

        self.backbone_name = backbone + '_fpn' if with_fpn else backbone

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

        elif 'ResNet50_vd' in backbone:
            if not with_fpn:
                logging.warning(
                    "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                    format(backbone))
                with_fpn = True
            backbone = self._get_backbone(
                'ResNet',
                variant='d',
                norm_type='bn',
                freeze_at=0,
                return_idx=[0, 1, 2, 3],
                num_stages=4,
                lr_mult_list=[0.05, 0.05, 0.1, 0.15]
                if '_ssld' in backbone else [1.0, 1.0, 1.0, 1.0])

        else:
            if not with_fpn:
                logging.warning(
                    "Backbone {} should be used along with fpn enabled, 'with_fpn' is forcibly set to True".
                    format(backbone))
                with_fpn = True
            backbone = self._get_backbone(
                'ResNet',
                variant='d' if '_vd' in backbone else 'b',
                depth=101,
                norm_type='bn',
                freeze_at=0,
                return_idx=[0, 1, 2, 3],
                num_stages=4)

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
            bb_head = heads.TwoFCHead(
                in_channel=neck.out_shape[0].channels, out_channel=1024)
            bb_roi_extractor_cfg = {
                'resolution': 7,
                'spatial_scale': [1. / i.stride for i in neck.out_shape],
                'sampling_ratio': 0,
                'aligned': True
            }
            with_pool = False
            m_head = heads.MaskFeat(
                in_channel=neck.out_shape[0].channels,
                out_channel=256,
                num_convs=4)
            m_roi_extractor_cfg = {
                'resolution': 14,
                'spatial_scale': [1. / i.stride for i in neck.out_shape],
                'sampling_ratio': 0,
                'aligned': True
            }
            mask_assigner = MaskAssigner(
                num_classes=num_classes, mask_resolution=28)
            share_bbox_feat = False

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
            bb_head = backbones.Res5Head()
            bb_roi_extractor_cfg = {
                'resolution': 14,
                'spatial_scale': [1. / i.stride for i in backbone.out_shape],
                'sampling_ratio': 0,
                'aligned': True
            }
            with_pool = True
            m_head = heads.MaskFeat(
                in_channel=bb_head.out_shape[0].channels,
                out_channel=256,
                num_convs=0)
            m_roi_extractor_cfg = {
                'resolution': 14,
                'spatial_scale': [1. / i.stride for i in backbone.out_shape],
                'sampling_ratio': 0,
                'aligned': True
            }
            mask_assigner = MaskAssigner(
                num_classes=num_classes, mask_resolution=14)
            share_bbox_feat = True

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
            head=bb_head,
            in_channel=bb_head.out_shape[0].channels,
            roi_extractor=bb_roi_extractor_cfg,
            with_pool=with_pool,
            bbox_assigner=bbox_assigner,
            num_classes=num_classes)

        mask_head = heads.MaskHead(
            head=m_head,
            roi_extractor=m_roi_extractor_cfg,
            mask_assigner=mask_assigner,
            share_bbox_feat=share_bbox_feat,
            num_classes=num_classes)

        bbox_post_process = BBoxPostProcess(
            num_classes=num_classes,
            decode=RCNNBox(num_classes=num_classes),
            nms=MultiClassNMS(
                score_threshold=score_threshold,
                keep_top_k=keep_top_k,
                nms_threshold=nms_threshold))

        mask_post_process = MaskPostProcess(binary_thresh=.5)

        params = {
            'backbone': backbone,
            'neck': neck,
            'rpn_head': rpn_head,
            'bbox_head': bbox_head,
            'mask_head': mask_head,
            'bbox_post_process': bbox_post_process,
            'mask_post_process': mask_post_process
        }
        self.with_fpn = with_fpn
        super(MaskRCNN, self).__init__(
            model_name='MaskRCNN', num_classes=num_classes, **params)

    def _compose_batch_transform(self, transforms, mode='train'):
        if mode == 'train':
            default_batch_transforms = [
                _BatchPadding(
                    pad_to_stride=32 if self.with_fpn else -1, pad_gt=True)
            ]
        else:
            default_batch_transforms = [
                _BatchPadding(
                    pad_to_stride=32 if self.with_fpn else -1, pad_gt=False)
            ]
        custom_batch_transforms = []
        for i, op in enumerate(transforms.transforms):
            if isinstance(op, (BatchRandomResize, BatchRandomResizeByShort)):
                custom_batch_transforms.insert(0, copy.deepcopy(op))

        batch_transforms = BatchCompose(custom_batch_transforms +
                                        default_batch_transforms)

        return batch_transforms
