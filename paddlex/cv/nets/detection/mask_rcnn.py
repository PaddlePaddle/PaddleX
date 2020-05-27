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

import paddle.fluid as fluid

from .fpn import (FPN, HRFPN)
from .rpn_head import (RPNHead, FPNRPNHead)
from .roi_extractor import (RoIAlign, FPNRoIAlign)
from .bbox_head import (BBoxHead, TwoFCHead)
from .mask_head import MaskHead
from ..resnet import ResNetC5

__all__ = ['MaskRCNN']


class MaskRCNN(object):
    """
    Mask R-CNN architecture, see https://arxiv.org/abs/1703.06870
    Args:
        backbone (object): backbone instance
        rpn_head (object): `RPNhead` instance
        roi_extractor (object): ROI extractor instance
        bbox_head (object): `BBoxHead` instance
        mask_head (object): `MaskHead` instance
        fpn (object): feature pyramid network instance
    """

    def __init__(
            self,
            backbone,
            num_classes=81,
            mode='train',
            with_fpn=False,
            fpn=None,
            num_chan=256,
            min_level=2,
            max_level=6,
            spatial_scale=[1. / 32., 1. / 16., 1. / 8., 1. / 4.],
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
            #roi_extractor
            roi_extractor=None,
            #bbox_head
            bbox_head=None,
            keep_top_k=100,
            nms_threshold=0.5,
            score_threshold=0.05,
            #MaskHead
            mask_head=None,
            num_convs=0,
            mask_head_resolution=14,
            #bbox_assigner
            batch_size_per_im=512,
            fg_fraction=.25,
            fg_thresh=.5,
            bg_thresh_hi=.5,
            bg_thresh_lo=0.,
            bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
            fixed_input_shape=None):
        super(MaskRCNN, self).__init__()
        self.backbone = backbone
        self.mode = mode
        if with_fpn and fpn is None:
            if self.backbone.__class__.__name__.startswith('HRNet'):
                fpn = HRFPN()
                fpn.min_level = 2
                fpn.max_level = 6
            else:
                fpn = FPN(num_chan=num_chan,
                          min_level=min_level,
                          max_level=max_level,
                          spatial_scale=spatial_scale)
        self.fpn = fpn
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
                    test_nms_thresh=test_nms_thresh)
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
                    test_nms_thresh=test_nms_thresh)
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
                    freeze_norm=self.backbone.freeze_norm)
            else:
                head = TwoFCHead()
            bbox_head = BBoxHead(
                head=head,
                keep_top_k=keep_top_k,
                nms_threshold=nms_threshold,
                score_threshold=score_threshold,
                num_classes=num_classes)
        self.bbox_head = bbox_head
        if mask_head is None:
            mask_head = MaskHead(
                num_convs=num_convs,
                resolution=mask_head_resolution,
                num_classes=num_classes)
        self.mask_head = mask_head
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.rpn_only = rpn_only
        self.fixed_input_shape = fixed_input_shape

    def build_net(self, inputs):
        im = inputs['image']
        im_info = inputs['im_info']

        # backbone
        body_feats = self.backbone(im)

        # FPN
        spatial_scale = None
        if self.fpn is not None:
            body_feats, spatial_scale = self.fpn.get_output(body_feats)

        # RPN proposals
        rois = self.rpn_head.get_proposals(body_feats, im_info, mode=self.mode)

        if self.mode == 'train':
            rpn_loss = self.rpn_head.get_loss(im_info, inputs['gt_box'],
                                              inputs['is_crowd'])
            outputs = fluid.layers.generate_proposal_labels(
                rpn_rois=rois,
                gt_classes=inputs['gt_label'],
                is_crowd=inputs['is_crowd'],
                gt_boxes=inputs['gt_box'],
                im_info=inputs['im_info'],
                batch_size_per_im=self.batch_size_per_im,
                fg_fraction=self.fg_fraction,
                fg_thresh=self.fg_thresh,
                bg_thresh_hi=self.bg_thresh_hi,
                bg_thresh_lo=self.bg_thresh_lo,
                bbox_reg_weights=self.bbox_reg_weights,
                class_nums=self.num_classes,
                use_random=self.rpn_head.use_random)

            rois = outputs[0]
            labels_int32 = outputs[1]

            if self.fpn is None:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)

            loss = self.bbox_head.get_loss(roi_feat, labels_int32,
                                           *outputs[2:])
            loss.update(rpn_loss)

            mask_rois, roi_has_mask_int32, mask_int32 = fluid.layers.generate_mask_labels(
                rois=rois,
                gt_classes=inputs['gt_label'],
                is_crowd=inputs['is_crowd'],
                gt_segms=inputs['gt_mask'],
                im_info=inputs['im_info'],
                labels_int32=labels_int32,
                num_classes=self.num_classes,
                resolution=self.mask_head.resolution)
            if self.fpn is None:
                bbox_head_feat = self.bbox_head.get_head_feat()
                feat = fluid.layers.gather(bbox_head_feat, roi_has_mask_int32)
            else:
                feat = self.roi_extractor(
                    body_feats, mask_rois, spatial_scale, is_mask=True)

            mask_loss = self.mask_head.get_loss(feat, mask_int32)
            loss.update(mask_loss)

            total_loss = fluid.layers.sum(list(loss.values()))
            loss.update({'loss': total_loss})
            return loss

        else:
            if self.rpn_only:
                im_scale = fluid.layers.slice(
                    im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, rois)
                rois = rois / im_scale
                return {'proposal': rois}
            mask_name = 'mask_pred'
            mask_pred, bbox_pred = self._eval(body_feats, mask_name, rois,
                                              im_info, inputs['im_shape'],
                                              spatial_scale)
            return OrderedDict(zip(['bbox', 'mask'], [bbox_pred, mask_pred]))

    def _eval(self,
              body_feats,
              mask_name,
              rois,
              im_info,
              im_shape,
              spatial_scale,
              bbox_pred=None):
        if not bbox_pred:
            if self.fpn is None:
                last_feat = body_feats[list(body_feats.keys())[-1]]
                roi_feat = self.roi_extractor(last_feat, rois)
            else:
                roi_feat = self.roi_extractor(body_feats, rois, spatial_scale)
            bbox_pred = self.bbox_head.get_prediction(roi_feat, rois, im_info,
                                                      im_shape)
            bbox_pred = bbox_pred['bbox']

        # share weight
        bbox_shape = fluid.layers.shape(bbox_pred)
        bbox_size = fluid.layers.reduce_prod(bbox_shape)
        bbox_size = fluid.layers.reshape(bbox_size, [1, 1])
        size = fluid.layers.fill_constant([1, 1], value=6, dtype='int32')
        cond = fluid.layers.less_than(x=bbox_size, y=size)

        mask_pred = fluid.layers.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=False,
            name=mask_name)
        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(cond):
                fluid.layers.assign(input=bbox_pred, output=mask_pred)
            with switch.default():
                bbox = fluid.layers.slice(bbox_pred, [1], starts=[2], ends=[6])

                im_scale = fluid.layers.slice(
                    im_info, [1], starts=[2], ends=[3])
                im_scale = fluid.layers.sequence_expand(im_scale, bbox)

                mask_rois = bbox * im_scale
                if self.fpn is None:
                    last_feat = body_feats[list(body_feats.keys())[-1]]
                    mask_feat = self.roi_extractor(last_feat, mask_rois)
                    mask_feat = self.bbox_head.get_head_feat(mask_feat)
                else:
                    mask_feat = self.roi_extractor(
                        body_feats, mask_rois, spatial_scale, is_mask=True)

                mask_out = self.mask_head.get_prediction(mask_feat, bbox)
                fluid.layers.assign(input=mask_out, output=mask_pred)
        return mask_pred, bbox_pred

    def generate_inputs(self):
        inputs = OrderedDict()

        if self.fixed_input_shape is not None:
            input_shape = [
                None, 3, self.fixed_input_shape[1], self.fixed_input_shape[0]
            ]
            inputs['image'] = fluid.data(
                dtype='float32', shape=input_shape, name='image')
        else:
            inputs['image'] = fluid.data(
                dtype='float32', shape=[None, 3, None, None], name='image')
        if self.mode == 'train':
            inputs['im_info'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_info')
            inputs['gt_box'] = fluid.data(
                dtype='float32', shape=[None, 4], lod_level=1, name='gt_box')
            inputs['gt_label'] = fluid.data(
                dtype='int32', shape=[None, 1], lod_level=1, name='gt_label')
            inputs['is_crowd'] = fluid.data(
                dtype='int32', shape=[None, 1], lod_level=1, name='is_crowd')
            inputs['gt_mask'] = fluid.data(
                dtype='float32', shape=[None, 2], lod_level=3, name='gt_mask')
        elif self.mode == 'eval':
            inputs['im_info'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_info')
            inputs['im_id'] = fluid.data(
                dtype='int64', shape=[None, 1], name='im_id')
            inputs['im_shape'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_shape')
        elif self.mode == 'test':
            inputs['im_info'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_info')
            inputs['im_shape'] = fluid.data(
                dtype='float32', shape=[None, 3], name='im_shape')
        return inputs
