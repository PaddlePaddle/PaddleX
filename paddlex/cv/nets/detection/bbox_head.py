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

from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, Xavier
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import MSRA


__all__ = ['BBoxHead', 'TwoFCHead']


class TwoFCHead(object):
    """
    RCNN head with two Fully Connected layers

    Args:
        mlp_dim (int): num of filters for the fc layers
    """

    def __init__(self, mlp_dim=1024):
        super(TwoFCHead, self).__init__()
        self.mlp_dim = mlp_dim

    def __call__(self, roi_feat):
        fan = roi_feat.shape[1] * roi_feat.shape[2] * roi_feat.shape[3]

        fc6 = fluid.layers.fc(
            input=roi_feat,
            size=self.mlp_dim,
            act='relu',
            name='fc6',
            param_attr=ParamAttr(
                name='fc6_w', initializer=Xavier(fan_out=fan)),
            bias_attr=ParamAttr(
                name='fc6_b', learning_rate=2., regularizer=L2Decay(0.)))
        head_feat = fluid.layers.fc(
            input=fc6,
            size=self.mlp_dim,
            act='relu',
            name='fc7',
            param_attr=ParamAttr(name='fc7_w', initializer=Xavier()),
            bias_attr=ParamAttr(
                name='fc7_b', learning_rate=2., regularizer=L2Decay(0.)))

        return head_feat


class BBoxHead(object):
    def __init__(
            self,
            head,
            #box_coder
            prior_box_var=[0.1, 0.1, 0.2, 0.2],
            code_type='decode_center_size',
            box_normalized=False,
            axis=1,
            #MultiClassNMS
            score_threshold=.05,
            nms_top_k=-1,
            keep_top_k=100,
            nms_threshold=.5,
            normalized=False,
            nms_eta=1.0,
            background_label=0,
            #bbox_loss
            sigma=1.0,
            num_classes=81,
            bbox_loss_type='SmoothL1Loss'):
        super(BBoxHead, self).__init__()
        self.head = head
        self.prior_box_var = prior_box_var
        self.code_type = code_type
        self.box_normalized = box_normalized
        self.axis = axis
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.background_label = background_label
        self.sigma = sigma
        self.num_classes = num_classes
        self.head_feat = None
        self.bbox_loss_type = bbox_loss_type

    def get_head_feat(self, input=None):
        """
        Get the bbox head feature map.
        """

        if input is not None:
            feat = self.head(input)
            if isinstance(feat, OrderedDict):
                feat = list(feat.values())[0]
            self.head_feat = feat
        return self.head_feat

    def _get_output(self, roi_feat):
        """
        Get bbox head output.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.

        Returns:
            cls_score(Variable): Output of rpn head with shape of
                [N, num_anchors, H, W].
            bbox_pred(Variable): Output of rpn head with shape of
                [N, num_anchors * 4, H, W].
        """
        head_feat = self.get_head_feat(roi_feat)
        
        # when ResNetC5 output a single feature map
        if not isinstance(self.head, TwoFCHead):
            head_feat = fluid.layers.pool2d(
                head_feat, pool_type='avg', global_pooling=True)
        cls_score = fluid.layers.fc(
            input=head_feat,
            size=self.num_classes,
            act=None,
            name='cls_score',
            param_attr=ParamAttr(
                name='cls_score_w', initializer=Normal(loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name='cls_score_b', learning_rate=2., regularizer=L2Decay(0.)))
        bbox_pred = fluid.layers.fc(
            input=head_feat,
            size=4 * self.num_classes,
            act=None,
            name='bbox_pred',
            param_attr=ParamAttr(
                name='bbox_pred_w', initializer=Normal(loc=0.0, scale=0.001)),
            bias_attr=ParamAttr(
                name='bbox_pred_b', learning_rate=2., regularizer=L2Decay(0.)))
        return cls_score, bbox_pred

    def get_loss(self, roi_feat, labels_int32, bbox_targets,
                 bbox_inside_weights, bbox_outside_weights):
        """
        Get bbox_head loss.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            labels_int32(Variable): Class label of a RoI with shape [P, 1].
                P is the number of RoI.
            bbox_targets(Variable): Box label of a RoI with shape
                [P, 4 * class_nums].
            bbox_inside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.
            bbox_outside_weights(Variable): Indicates whether a box should
                contribute to loss. Same shape as bbox_targets.

        Return:
            Type: Dict
                loss_cls(Variable): bbox_head loss.
                loss_bbox(Variable): bbox_head loss.
        """

        cls_score, bbox_pred = self._get_output(roi_feat)
        labels_int64 = fluid.layers.cast(x=labels_int32, dtype='int64')
        labels_int64.stop_gradient = True
        loss_cls = fluid.layers.softmax_with_cross_entropy(
            logits=cls_score, label=labels_int64, numeric_stable_mode=True)
        loss_cls = fluid.layers.reduce_mean(loss_cls)
        if self.bbox_loss_type == 'CiouLoss':
            from .loss.diou_loss import DiouLoss
            loss_obj = DiouLoss(loss_weight=10.,
                                is_cls_agnostic=False,
                                num_classes=self.num_classes,
                                use_complete_iou_loss=True)
        elif self.bbox_loss_type == 'DiouLoss':
            from .loss.diou_loss import DiouLoss
            loss_obj = DiouLoss(loss_weight=12.,
                                is_cls_agnostic=False,
                                num_classes=self.num_classes,
                                use_complete_iou_loss=False)
        elif self.bbox_loss_type == 'GiouLoss':
            from .loss.giou_loss import GiouLoss
            loss_obj = GiouLoss(loss_weight=10.,
                                is_cls_agnostic=False,
                                num_classes=self.num_classes)
        else:
            from .loss.smoothl1_loss import SmoothL1Loss
            loss_obj = SmoothL1Loss(self.sigma)
        loss_bbox = loss_obj(
                x=bbox_pred,
                y=bbox_targets,
                inside_weight=bbox_inside_weights,
                outside_weight=bbox_outside_weights)
        loss_bbox = fluid.layers.reduce_mean(loss_bbox)
        return {'loss_cls': loss_cls, 'loss_bbox': loss_bbox}

    def get_prediction(self,
                       roi_feat,
                       rois,
                       im_info,
                       im_shape,
                       return_box_score=False):
        """
        Get prediction bounding box in test stage.

        Args:
            roi_feat (Variable): RoI feature from RoIExtractor.
            rois (Variable): Output of generate_proposals in rpn head.
            im_info (Variable): A 2-D LoDTensor with shape [B, 3]. B is the
                number of input images, each element consists of im_height,
                im_width, im_scale.
            im_shape (Variable): Actual shape of original image with shape
                [B, 3]. B is the number of images, each element consists of
                original_height, original_width, 1

        Returns:
            pred_result(Variable): Prediction result with shape [N, 6]. Each
                row has 6 values: [label, confidence, xmin, ymin, xmax, ymax].
                N is the total number of prediction.
        """
        cls_score, bbox_pred = self._get_output(roi_feat)

        im_scale = fluid.layers.slice(im_info, [1], starts=[2], ends=[3])
        im_scale = fluid.layers.sequence_expand(im_scale, rois)
        boxes = rois / im_scale
        cls_prob = fluid.layers.softmax(cls_score, use_cudnn=False)
        bbox_pred = fluid.layers.reshape(bbox_pred, (-1, self.num_classes, 4))
        decoded_box = fluid.layers.box_coder(
            prior_box=boxes,
            target_box=bbox_pred,
            prior_box_var=self.prior_box_var,
            code_type=self.code_type,
            box_normalized=self.box_normalized,
            axis=self.axis)
        cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=im_shape)
        if return_box_score:
            return {'bbox': cliped_box, 'score': cls_prob}
        if self.bbox_loss_type == 'CiouLoss':
            from .nms import MultiClassDiouNMS
            nms_obj = MultiClassDiouNMS(score_threshold=self.score_threshold,
                                        nms_threshold=self.nms_threshold,
                                        keep_top_k=self.keep_top_k)
            pred_result = nms_obj(bboxes=cliped_box, scores=cls_prob)
        else:
            pred_result = fluid.layers.multiclass_nms(
                bboxes=cliped_box,
                scores=cls_prob,
                score_threshold=self.score_threshold,
                nms_top_k=self.nms_top_k,
                keep_top_k=self.keep_top_k,
                nms_threshold=self.nms_threshold,
                normalized=self.normalized,
                nms_eta=self.nms_eta,
                background_label=self.background_label)
        return {'bbox': pred_result}
