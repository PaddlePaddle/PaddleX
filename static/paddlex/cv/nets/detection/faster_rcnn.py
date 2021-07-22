# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import copy

from paddle import fluid

from .fpn import (FPN, HRFPN)
from .rpn_head import (RPNHead, FPNRPNHead)
from .roi_extractor import (RoIAlign, FPNRoIAlign)
from .bbox_head import (BBoxHead, TwoFCHead)
from ..resnet import ResNetC5
from .loss.diou_loss import DiouLoss
from .ops import BBoxAssigner, LibraBBoxAssigner

__all__ = ['FasterRCNN']


class FasterRCNN(object):
    """
    Faster R-CNN architecture, see https://arxiv.org/abs/1506.01497
    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNhead` instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `BBoxHead` instance
        fpn (object): feature pyramid network instance
    """

    def __init__(
            self,
            backbone,
            input_channel=3,
            mode='train',
            num_classes=81,
            with_fpn=False,
            fpn=None,
            #rpn_head
            rpn_only=False,
            rpn_head=None,
            anchor_sizes=[32, 64, 128, 256, 512],
            aspect_ratios=[0.5, 1.0, 2.0],
            rpn_batch_size_per_im=256,
            rpn_fg_fraction=0.5,
            rpn_positive_overlap=0.7,
            rpn_negative_overlap=0.3,
            train_pre_nms_top_n=12000,
            train_post_nms_top_n=2000,
            train_nms_thresh=0.7,
            test_pre_nms_top_n=6000,
            test_post_nms_top_n=1000,
            test_nms_thresh=0.7,
            rpn_cls_loss='SigmoidCrossEntropy',
            rpn_focal_loss_alpha=0.25,
            rpn_focal_loss_gamma=2,
            #roi_extractor
            roi_extractor=None,
            #bbox_head
            bbox_head=None,
            keep_top_k=100,
            nms_threshold=0.5,
            score_threshold=0.05,
            rcnn_nms='MultiClassNMS',
            softnms_sigma=0.5,
            post_threshold=.05,
            #bbox_assigner
            batch_size_per_im=512,
            fg_fraction=.25,
            fg_thresh=.5,
            bg_thresh_hi=.5,
            bg_thresh_lo=0.,
            bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
            fixed_input_shape=None,
            rcnn_bbox_loss='SmoothL1Loss',
            diouloss_weight=10.0,
            diouloss_is_cls_agnostic=False,
            diouloss_use_complete_iou_loss=True,
            bbox_assigner='BBoxAssigner',
            fpn_num_channels=256):
        super(FasterRCNN, self).__init__()
        self.backbone = backbone
        self.mode = mode
        if with_fpn and fpn is None:
            if self.backbone.__class__.__name__.startswith('HRNet'):
                fpn = HRFPN()
                fpn.min_level = 2
                fpn.max_level = 6
            else:
                fpn = FPN()
        self.fpn = fpn
        if self.fpn is not None:
            self.fpn.num_chan = fpn_num_channels
        self.num_classes = num_classes
        if rpn_head is None:
            if self.fpn is None:
                rpn_head = RPNHead(
                    anchor_sizes=anchor_sizes,
                    aspect_ratios=aspect_ratios,
                    rpn_batch_size_per_im=rpn_batch_size_per_im,
                    rpn_fg_fraction=rpn_fg_fraction,
                    rpn_positive_overlap=rpn_positive_overlap,
                    rpn_negative_overlap=rpn_negative_overlap,
                    train_pre_nms_top_n=train_pre_nms_top_n,
                    train_post_nms_top_n=train_post_nms_top_n,
                    train_nms_thresh=train_nms_thresh,
                    test_pre_nms_top_n=test_pre_nms_top_n,
                    test_post_nms_top_n=test_post_nms_top_n,
                    test_nms_thresh=test_nms_thresh,
                    rpn_cls_loss=rpn_cls_loss,
                    rpn_focal_loss_alpha=rpn_focal_loss_alpha,
                    rpn_focal_loss_gamma=rpn_focal_loss_gamma)
            else:
                rpn_head = FPNRPNHead(
                    anchor_start_size=anchor_sizes[0],
                    aspect_ratios=aspect_ratios,
                    num_chan=self.fpn.num_chan,
                    min_level=self.fpn.min_level,
                    max_level=self.fpn.max_level,
                    rpn_batch_size_per_im=rpn_batch_size_per_im,
                    rpn_fg_fraction=rpn_fg_fraction,
                    rpn_positive_overlap=rpn_positive_overlap,
                    rpn_negative_overlap=rpn_negative_overlap,
                    train_pre_nms_top_n=train_pre_nms_top_n,
                    train_post_nms_top_n=train_post_nms_top_n,
                    train_nms_thresh=train_nms_thresh,
                    test_pre_nms_top_n=test_pre_nms_top_n,
                    test_post_nms_top_n=test_post_nms_top_n,
                    test_nms_thresh=test_nms_thresh,
                    rpn_cls_loss=rpn_cls_loss,
                    rpn_focal_loss_alpha=rpn_focal_loss_alpha,
                    rpn_focal_loss_gamma=rpn_focal_loss_gamma)
        self.rpn_head = rpn_head
        if roi_extractor is None:
            if self.fpn is None:
                roi_extractor = RoIAlign(
                    resolution=14,
                    spatial_scale=1. / 2**self.backbone.feature_maps[0])
            else:
                roi_extractor = FPNRoIAlign(sampling_ratio=2)
        self.roi_extractor = roi_extractor
        if bbox_head is None:
            if self.fpn is None:
                head = ResNetC5(
                    layers=self.backbone.layers,
                    norm_type=self.backbone.norm_type,
                    freeze_norm=self.backbone.freeze_norm,
                    variant=self.backbone.variant)
            else:
                head = TwoFCHead()
            bbox_head = BBoxHead(
                head=head,
                keep_top_k=keep_top_k,
                nms_threshold=nms_threshold,
                score_threshold=score_threshold,
                rcnn_nms=rcnn_nms,
                softnms_sigma=softnms_sigma,
                post_threshold=post_threshold,
                num_classes=num_classes,
                rcnn_bbox_loss=rcnn_bbox_loss,
                diouloss_weight=diouloss_weight,
                diouloss_is_cls_agnostic=diouloss_is_cls_agnostic,
                diouloss_use_complete_iou_loss=diouloss_use_complete_iou_loss)

        self.bbox_head = bbox_head
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.rpn_only = rpn_only
        self.fixed_input_shape = fixed_input_shape
        if bbox_assigner == 'BBoxAssigner':
            self.bbox_assigner = BBoxAssigner(
                batch_size_per_im=batch_size_per_im,
                fg_fraction=fg_fraction,
                fg_thresh=fg_thresh,
                bg_thresh_hi=bg_thresh_hi,
                bg_thresh_lo=bg_thresh_lo,
                bbox_reg_weights=bbox_reg_weights,
                num_classes=num_classes,
                shuffle_before_sample=self.rpn_head.use_random)
        elif bbox_assigner == 'LibraBBoxAssigner':
            self.bbox_assigner = LibraBBoxAssigner(
                batch_size_per_im=batch_size_per_im,
                fg_fraction=fg_fraction,
                fg_thresh=fg_thresh,
                bg_thresh_hi=bg_thresh_hi,
                bg_thresh_lo=bg_thresh_lo,
                bbox_reg_weights=bbox_reg_weights,
                num_classes=num_classes,
                shuffle_before_sample=self.rpn_head.use_random)
        self.input_channel = input_channel

    def build_net(self, inputs):
        im = inputs['image']
        im_info = inputs['im_info']
        if self.mode == 'train':
            gt_bbox = inputs['gt_box']
            is_crowd = inputs['is_crowd']
        else:
            im_shape = inputs['im_shape']

        body_feats = self.backbone(im)
        body_feat_names = list(body_feats.keys())

        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        rois = self.rpn_head.get_proposals(body_feats, im_info, mode=self.mode)

        if self.mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, gt_bbox, is_crowd)
            outputs = self.bbox_assigner(
                rpn_rois=rois,
                gt_classes=inputs['gt_label'],
                is_crowd=inputs['is_crowd'],
                gt_boxes=inputs['gt_box'],
                im_info=inputs['im_info'])

            rois = outputs[0]
            labels_int32 = outputs[1]
            bbox_targets = outputs[2]
            bbox_inside_weights = outputs[3]
            bbox_outside_weights = outputs[4]
        else:
            if self.rpn_only:
                im_scale = fluid.layers.slice(
                    im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, rois)
                rois = rois / im_scale
                return {'proposal': rois}
        if self.fpn is None:
            # in models without FPN, roi extractor only uses the last level of
            # feature maps. And body_feat_names[-1] represents the name of
            # last feature map.
            body_feat = body_feats[body_feat_names[-1]]
            roi_feat = self.roi_extractor(body_feat, rois)
        else:
            roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

        if self.mode == 'train':
            loss = self.bbox_head.get_loss(roi_feat, labels_int32,
                                           bbox_targets, bbox_inside_weights,
                                           bbox_outside_weights)
            loss.update(rpn_loss)
            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss
        else:
            pred = self.bbox_head.get_prediction(roi_feat, rois, im_info,
                                                 im_shape)
            return pred

    def generate_inputs(self):
        inputs = OrderedDict()

        if self.fixed_input_shape is not None:
            input_shape = [
                None, self.input_channel, self.fixed_input_shape[1],
                self.fixed_input_shape[0]
            ]
            inputs['image'] = fluid.data(
                dtype='float32', shape=input_shape, name='image')
        else:
            inputs['image'] = fluid.data(
                dtype='float32',
                shape=[None, self.input_channel, None, None],
                name='image')
        if self.mode == 'train':
            inputs['im_info'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_info')
            inputs['gt_box'] = fluid.data(
                dtype='float32', shape=[None, 4], lod_level=1, name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, 1], lod_level=1, name='gt_label')
            inputs['is_crowd'] = fluid.data(
                dtype='int32', shape=[None, 1], lod_level=1, name='is_crowd')
        elif self.mode == 'eval':
            inputs['im_info'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_info')
            inputs['im_id'] = fluid.data(
                dtype='int64', shape=[None, 1], name='im_id')
            inputs['im_shape'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_shape')
            inputs['gt_box'] = fluid.data(
                dtype='float32', shape=[None, 4], lod_level=1, name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, 1], lod_level=1, name='gt_label')
            inputs['is_difficult'] = fluid.data(
                dtype='int32',
                shape=[None, 1],
                lod_level=1,
                name='is_difficult')
        elif self.mode == 'test':
            inputs['im_info'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_info')
            inputs['im_shape'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_shape')
        return inputs
