# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from numbers import Integral
import math
import six

import paddle
from paddle import fluid


def bbox_overlaps(boxes_1, boxes_2):
    '''
    bbox_overlaps
        boxes_1: x1, y, x2, y2
        boxes_2: x1, y, x2, y2
    '''
    assert boxes_1.shape[1] == 4 and boxes_2.shape[1] == 4

    num_1 = boxes_1.shape[0]
    num_2 = boxes_2.shape[0]

    x1_1 = boxes_1[:, 0:1]
    y1_1 = boxes_1[:, 1:2]
    x2_1 = boxes_1[:, 2:3]
    y2_1 = boxes_1[:, 3:4]
    area_1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)

    x1_2 = boxes_2[:, 0].transpose()
    y1_2 = boxes_2[:, 1].transpose()
    x2_2 = boxes_2[:, 2].transpose()
    y2_2 = boxes_2[:, 3].transpose()
    area_2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

    xx1 = np.maximum(x1_1, x1_2)
    yy1 = np.maximum(y1_1, y1_2)
    xx2 = np.minimum(x2_1, x2_2)
    yy2 = np.minimum(y2_1, y2_2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    ovr = inter / (area_1 + area_2 - inter)
    return ovr


def box_to_delta(ex_boxes, gt_boxes, weights):
    """ box_to_delta """
    ex_w = ex_boxes[:, 2] - ex_boxes[:, 0] + 1
    ex_h = ex_boxes[:, 3] - ex_boxes[:, 1] + 1
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_w
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    targets = np.vstack([dx, dy, dw, dh]).transpose()
    return targets


def DropBlock(input, block_size, keep_prob, is_test):
    if is_test:
        return input

    def CalculateGamma(input, block_size, keep_prob):
        input_shape = fluid.layers.shape(input)
        feat_shape_tmp = fluid.layers.slice(input_shape, [0], [3], [4])
        feat_shape_tmp = fluid.layers.cast(feat_shape_tmp, dtype="float32")
        feat_shape_t = fluid.layers.reshape(feat_shape_tmp, [1, 1, 1, 1])
        feat_area = fluid.layers.pow(feat_shape_t, factor=2)

        block_shape_t = fluid.layers.fill_constant(
            shape=[1, 1, 1, 1], value=block_size, dtype='float32')
        block_area = fluid.layers.pow(block_shape_t, factor=2)

        useful_shape_t = feat_shape_t - block_shape_t + 1
        useful_area = fluid.layers.pow(useful_shape_t, factor=2)

        upper_t = feat_area * (1 - keep_prob)
        bottom_t = block_area * useful_area
        output = upper_t / bottom_t
        return output

    gamma = CalculateGamma(input, block_size=block_size, keep_prob=keep_prob)
    input_shape = fluid.layers.shape(input)
    p = fluid.layers.expand_as(gamma, input)

    input_shape_tmp = fluid.layers.cast(input_shape, dtype="int64")
    random_matrix = fluid.layers.uniform_random(
        input_shape_tmp, dtype='float32', min=0.0, max=1.0)
    one_zero_m = fluid.layers.less_than(random_matrix, p)
    one_zero_m.stop_gradient = True
    one_zero_m = fluid.layers.cast(one_zero_m, dtype="float32")

    mask_flag = fluid.layers.pool2d(
        one_zero_m,
        pool_size=block_size,
        pool_type='max',
        pool_stride=1,
        pool_padding=block_size // 2)
    mask = 1.0 - mask_flag

    elem_numel = fluid.layers.reduce_prod(input_shape)
    elem_numel_m = fluid.layers.cast(elem_numel, dtype="float32")
    elem_numel_m.stop_gradient = True

    elem_sum = fluid.layers.reduce_sum(mask)
    elem_sum_m = fluid.layers.cast(elem_sum, dtype="float32")
    elem_sum_m.stop_gradient = True

    output = input * mask * elem_numel_m / elem_sum_m
    return output


class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=False,
                 nms_eta=1.0,
                 background_label=0):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.background_label = background_label

    def __call__(self, bboxes, scores):
        return fluid.layers.multiclass_nms(
            bboxes=bboxes,
            scores=scores,
            score_threshold=self.score_threshold,
            nms_top_k=self.nms_top_k,
            keep_top_k=self.keep_top_k,
            normalized=self.normalized,
            nms_threshold=self.nms_threshold,
            nms_eta=self.nms_eta,
            background_label=self.background_label)


class MatrixNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 post_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 use_gaussian=False,
                 gaussian_sigma=2.,
                 normalized=False,
                 background_label=0):
        super(MatrixNMS, self).__init__()
        self.score_threshold = score_threshold
        self.post_threshold = post_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.normalized = normalized
        self.use_gaussian = use_gaussian
        self.gaussian_sigma = gaussian_sigma
        self.background_label = background_label

    def __call__(self, bboxes, scores):
        return paddle.fluid.layers.matrix_nms(
            bboxes=bboxes,
            scores=scores,
            score_threshold=self.score_threshold,
            post_threshold=self.post_threshold,
            nms_top_k=self.nms_top_k,
            keep_top_k=self.keep_top_k,
            normalized=self.normalized,
            use_gaussian=self.use_gaussian,
            gaussian_sigma=self.gaussian_sigma,
            background_label=self.background_label)


class MultiClassSoftNMS(object):
    def __init__(
            self,
            score_threshold=0.01,
            keep_top_k=300,
            softnms_sigma=0.5,
            normalized=False,
            background_label=0, ):
        super(MultiClassSoftNMS, self).__init__()
        self.score_threshold = score_threshold
        self.keep_top_k = keep_top_k
        self.softnms_sigma = softnms_sigma
        self.normalized = normalized
        self.background_label = background_label

    def __call__(self, bboxes, scores):
        def create_tmp_var(program, name, dtype, shape, lod_level):
            return program.current_block().create_var(
                name=name, dtype=dtype, shape=shape, lod_level=lod_level)

        def _soft_nms_for_cls(dets, sigma, thres):
            """soft_nms_for_cls"""
            dets_final = []
            while len(dets) > 0:
                maxpos = np.argmax(dets[:, 0])
                dets_final.append(dets[maxpos].copy())
                ts, tx1, ty1, tx2, ty2 = dets[maxpos]
                scores = dets[:, 0]
                # force remove bbox at maxpos
                scores[maxpos] = -1
                x1 = dets[:, 1]
                y1 = dets[:, 2]
                x2 = dets[:, 3]
                y2 = dets[:, 4]
                eta = 0 if self.normalized else 1
                areas = (x2 - x1 + eta) * (y2 - y1 + eta)
                xx1 = np.maximum(tx1, x1)
                yy1 = np.maximum(ty1, y1)
                xx2 = np.minimum(tx2, x2)
                yy2 = np.minimum(ty2, y2)
                w = np.maximum(0.0, xx2 - xx1 + eta)
                h = np.maximum(0.0, yy2 - yy1 + eta)
                inter = w * h
                ovr = inter / (areas + areas[maxpos] - inter)
                weight = np.exp(-(ovr * ovr) / sigma)
                scores = scores * weight
                idx_keep = np.where(scores >= thres)
                dets[:, 0] = scores
                dets = dets[idx_keep]
            dets_final = np.array(dets_final).reshape(-1, 5)
            return dets_final

        def _soft_nms(bboxes, scores):
            class_nums = scores.shape[-1]

            softnms_thres = self.score_threshold
            softnms_sigma = self.softnms_sigma
            keep_top_k = self.keep_top_k

            cls_boxes = [[] for _ in range(class_nums)]
            cls_ids = [[] for _ in range(class_nums)]

            start_idx = 1 if self.background_label == 0 else 0
            for j in range(start_idx, class_nums):
                inds = np.where(scores[:, j] >= softnms_thres)[0]
                scores_j = scores[inds, j]
                rois_j = bboxes[inds, j, :] if len(
                    bboxes.shape) > 2 else bboxes[inds, :]
                dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                    np.float32, copy=False)
                cls_rank = np.argsort(-dets_j[:, 0])
                dets_j = dets_j[cls_rank]

                cls_boxes[j] = _soft_nms_for_cls(
                    dets_j, sigma=softnms_sigma, thres=softnms_thres)
                cls_ids[j] = np.array([j] * cls_boxes[j].shape[0]).reshape(-1,
                                                                           1)

            cls_boxes = np.vstack(cls_boxes[start_idx:])
            cls_ids = np.vstack(cls_ids[start_idx:])
            pred_result = np.hstack([cls_ids, cls_boxes])

            # Limit to max_per_image detections **over all classes**
            image_scores = cls_boxes[:, 0]
            if len(image_scores) > keep_top_k:
                image_thresh = np.sort(image_scores)[-keep_top_k]
                keep = np.where(cls_boxes[:, 0] >= image_thresh)[0]
                pred_result = pred_result[keep, :]

            return pred_result

        def _batch_softnms(bboxes, scores):
            batch_offsets = bboxes.lod()
            bboxes = np.array(bboxes)
            scores = np.array(scores)
            out_offsets = [0]
            pred_res = []
            if len(batch_offsets) > 0:
                batch_offset = batch_offsets[0]
                for i in range(len(batch_offset) - 1):
                    s, e = batch_offset[i], batch_offset[i + 1]
                    pred = _soft_nms(bboxes[s:e], scores[s:e])
                    out_offsets.append(pred.shape[0] + out_offsets[-1])
                    pred_res.append(pred)
            else:
                assert len(bboxes.shape) == 3
                assert len(scores.shape) == 3
                for i in range(bboxes.shape[0]):
                    pred = _soft_nms(bboxes[i], scores[i])
                    out_offsets.append(pred.shape[0] + out_offsets[-1])
                    pred_res.append(pred)

            res = fluid.LoDTensor()
            res.set_lod([out_offsets])
            if len(pred_res) == 0:
                pred_res = np.array([[1]], dtype=np.float32)
            res.set(np.vstack(pred_res).astype(np.float32), fluid.CPUPlace())
            return res

        pred_result = create_tmp_var(
            fluid.default_main_program(),
            name='softnms_pred_result',
            dtype='float32',
            shape=[-1, 6],
            lod_level=1)
        fluid.layers.py_func(
            func=_batch_softnms, x=[bboxes, scores], out=pred_result)
        return pred_result


class MultiClassDiouNMS(object):
    def __init__(
            self,
            score_threshold=0.05,
            keep_top_k=100,
            nms_threshold=0.5,
            normalized=False,
            background_label=0, ):
        super(MultiClassDiouNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.keep_top_k = keep_top_k
        self.normalized = normalized
        self.background_label = background_label

    def __call__(self, bboxes, scores):
        def create_tmp_var(program, name, dtype, shape, lod_level):
            return program.current_block().create_var(
                name=name, dtype=dtype, shape=shape, lod_level=lod_level)

        def _calc_diou_term(dets1, dets2):
            eps = 1.e-10
            eta = 0 if self.normalized else 1

            x1, y1, x2, y2 = dets1[0], dets1[1], dets1[2], dets1[3]
            x1g, y1g, x2g, y2g = dets2[0], dets2[1], dets2[2], dets2[3]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1 + eta
            h = y2 - y1 + eta

            cxg = (x1g + x2g) / 2
            cyg = (y1g + y2g) / 2
            wg = x2g - x1g + eta
            hg = y2g - y1g + eta

            x2 = np.maximum(x1, x2)
            y2 = np.maximum(y1, y2)

            # A or B
            xc1 = np.minimum(x1, x1g)
            yc1 = np.minimum(y1, y1g)
            xc2 = np.maximum(x2, x2g)
            yc2 = np.maximum(y2, y2g)

            # DIOU term
            dist_intersection = (cx - cxg)**2 + (cy - cyg)**2
            dist_union = (xc2 - xc1)**2 + (yc2 - yc1)**2
            diou_term = (dist_intersection + eps) / (dist_union + eps)
            return diou_term

        def _diou_nms_for_cls(dets, thres):
            """_diou_nms_for_cls"""
            scores = dets[:, 0]
            x1 = dets[:, 1]
            y1 = dets[:, 2]
            x2 = dets[:, 3]
            y2 = dets[:, 4]
            eta = 0 if self.normalized else 1
            areas = (x2 - x1 + eta) * (y2 - y1 + eta)
            dt_num = dets.shape[0]
            order = np.array(range(dt_num))

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1 + eta)
                h = np.maximum(0.0, yy2 - yy1 + eta)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                diou_term = _calc_diou_term([x1[i], y1[i], x2[i], y2[i]], [
                    x1[order[1:]], y1[order[1:]], x2[order[1:]], y2[order[1:]]
                ])

                inds = np.where(ovr - diou_term <= thres)[0]

                order = order[inds + 1]

            dets_final = dets[keep]
            return dets_final

        def _diou_nms(bboxes, scores):
            bboxes = np.array(bboxes)
            scores = np.array(scores)
            class_nums = scores.shape[-1]

            score_threshold = self.score_threshold
            nms_threshold = self.nms_threshold
            keep_top_k = self.keep_top_k

            cls_boxes = [[] for _ in range(class_nums)]
            cls_ids = [[] for _ in range(class_nums)]

            start_idx = 1 if self.background_label == 0 else 0
            for j in range(start_idx, class_nums):
                inds = np.where(scores[:, j] >= score_threshold)[0]
                scores_j = scores[inds, j]
                rois_j = bboxes[inds, j, :]
                dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(
                    np.float32, copy=False)
                cls_rank = np.argsort(-dets_j[:, 0])
                dets_j = dets_j[cls_rank]

                cls_boxes[j] = _diou_nms_for_cls(dets_j, thres=nms_threshold)
                cls_ids[j] = np.array([j] * cls_boxes[j].shape[0]).reshape(-1,
                                                                           1)

            cls_boxes = np.vstack(cls_boxes[start_idx:])
            cls_ids = np.vstack(cls_ids[start_idx:])
            pred_result = np.hstack([cls_ids, cls_boxes]).astype(np.float32)

            # Limit to max_per_image detections **over all classes**
            image_scores = cls_boxes[:, 0]
            if len(image_scores) > keep_top_k:
                image_thresh = np.sort(image_scores)[-keep_top_k]
                keep = np.where(cls_boxes[:, 0] >= image_thresh)[0]
                pred_result = pred_result[keep, :]

            res = fluid.LoDTensor()
            res.set_lod([[0, pred_result.shape[0]]])
            if pred_result.shape[0] == 0:
                pred_result = np.array([[1]], dtype=np.float32)
            res.set(pred_result, fluid.CPUPlace())

            return res

        pred_result = create_tmp_var(
            fluid.default_main_program(),
            name='diou_nms_pred_result',
            dtype='float32',
            shape=[-1, 6],
            lod_level=0)
        fluid.layers.py_func(
            func=_diou_nms, x=[bboxes, scores], out=pred_result)
        return pred_result


class LibraBBoxAssigner(object):
    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh_hi=.5,
                 bg_thresh_lo=0.,
                 bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                 num_classes=81,
                 shuffle_before_sample=True,
                 is_cls_agnostic=False,
                 num_bins=3):
        super(LibraBBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.class_nums = num_classes
        self.use_random = shuffle_before_sample
        self.is_cls_agnostic = is_cls_agnostic
        self.num_bins = num_bins

    def __call__(
            self,
            rpn_rois,
            gt_classes,
            is_crowd,
            gt_boxes,
            im_info, ):
        return self.generate_proposal_label_libra(
            rpn_rois=rpn_rois,
            gt_classes=gt_classes,
            is_crowd=is_crowd,
            gt_boxes=gt_boxes,
            im_info=im_info,
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self.fg_fraction,
            fg_thresh=self.fg_thresh,
            bg_thresh_hi=self.bg_thresh_hi,
            bg_thresh_lo=self.bg_thresh_lo,
            bbox_reg_weights=self.bbox_reg_weights,
            class_nums=self.class_nums,
            use_random=self.use_random,
            is_cls_agnostic=self.is_cls_agnostic,
            is_cascade_rcnn=False)

    def generate_proposal_label_libra(
            self, rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
            batch_size_per_im, fg_fraction, fg_thresh, bg_thresh_hi,
            bg_thresh_lo, bbox_reg_weights, class_nums, use_random,
            is_cls_agnostic, is_cascade_rcnn):
        num_bins = self.num_bins

        def create_tmp_var(program, name, dtype, shape, lod_level=None):
            return program.current_block().create_var(
                name=name, dtype=dtype, shape=shape, lod_level=lod_level)

        def _sample_pos(max_overlaps, max_classes, pos_inds, num_expected):
            if len(pos_inds) <= num_expected:
                return pos_inds
            else:
                unique_gt_inds = np.unique(max_classes[pos_inds])
                num_gts = len(unique_gt_inds)
                num_per_gt = int(round(num_expected / float(num_gts)) + 1)

                sampled_inds = []
                for i in unique_gt_inds:
                    inds = np.nonzero(max_classes == i)[0]
                    before_len = len(inds)
                    inds = list(set(inds) & set(pos_inds))
                    after_len = len(inds)
                    if len(inds) > num_per_gt:
                        inds = np.random.choice(
                            inds, size=num_per_gt, replace=False)
                    sampled_inds.extend(list(inds))  # combine as a new sampler
                if len(sampled_inds) < num_expected:
                    num_extra = num_expected - len(sampled_inds)
                    extra_inds = np.array(
                        list(set(pos_inds) - set(sampled_inds)))
                    assert len(sampled_inds)+len(extra_inds) == len(pos_inds), \
                        "sum of sampled_inds({}) and extra_inds({}) length must be equal with pos_inds({})!".format(
                            len(sampled_inds), len(extra_inds), len(pos_inds))
                    if len(extra_inds) > num_extra:
                        extra_inds = np.random.choice(
                            extra_inds, size=num_extra, replace=False)
                    sampled_inds.extend(extra_inds.tolist())
                elif len(sampled_inds) > num_expected:
                    sampled_inds = np.random.choice(
                        sampled_inds, size=num_expected, replace=False)
                return sampled_inds

        def sample_via_interval(max_overlaps, full_set, num_expected,
                                floor_thr, num_bins, bg_thresh_hi):
            max_iou = max_overlaps.max()
            iou_interval = (max_iou - floor_thr) / num_bins
            per_num_expected = int(num_expected / num_bins)

            sampled_inds = []
            for i in range(num_bins):
                start_iou = floor_thr + i * iou_interval
                end_iou = floor_thr + (i + 1) * iou_interval

                tmp_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= start_iou, max_overlaps
                                       < end_iou))[0])
                tmp_inds = list(tmp_set & full_set)

                if len(tmp_inds) > per_num_expected:
                    tmp_sampled_set = np.random.choice(
                        tmp_inds, size=per_num_expected, replace=False)
                else:
                    tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
                sampled_inds.append(tmp_sampled_set)

            sampled_inds = np.concatenate(sampled_inds)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(full_set - set(sampled_inds)))
                assert len(sampled_inds)+len(extra_inds) == len(full_set), \
                    "sum of sampled_inds({}) and extra_inds({}) length must be equal with full_set({})!".format(
                            len(sampled_inds), len(extra_inds), len(full_set))

                if len(extra_inds) > num_extra:
                    extra_inds = np.random.choice(
                        extra_inds, num_extra, replace=False)
                sampled_inds = np.concatenate([sampled_inds, extra_inds])

            return sampled_inds

        def _sample_neg(max_overlaps,
                        max_classes,
                        neg_inds,
                        num_expected,
                        floor_thr=-1,
                        floor_fraction=0,
                        num_bins=3,
                        bg_thresh_hi=0.5):
            if len(neg_inds) <= num_expected:
                return neg_inds
            else:
                # balance sampling for negative samples
                neg_set = set(neg_inds)
                if floor_thr > 0:
                    floor_set = set(
                        np.where(
                            np.logical_and(max_overlaps >= 0, max_overlaps <
                                           floor_thr))[0])
                    iou_sampling_set = set(
                        np.where(max_overlaps >= floor_thr)[0])
                elif floor_thr == 0:
                    floor_set = set(np.where(max_overlaps == 0)[0])
                    iou_sampling_set = set(
                        np.where(max_overlaps > floor_thr)[0])
                else:
                    floor_set = set()
                    iou_sampling_set = set(
                        np.where(max_overlaps > floor_thr)[0])
                    floor_thr = 0

                floor_neg_inds = list(floor_set & neg_set)
                iou_sampling_neg_inds = list(iou_sampling_set & neg_set)

                num_expected_iou_sampling = int(num_expected *
                                                (1 - floor_fraction))
                if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                    if num_bins >= 2:
                        iou_sampled_inds = sample_via_interval(
                            max_overlaps,
                            set(iou_sampling_neg_inds),
                            num_expected_iou_sampling, floor_thr, num_bins,
                            bg_thresh_hi)
                    else:
                        iou_sampled_inds = np.random.choice(
                            iou_sampling_neg_inds,
                            size=num_expected_iou_sampling,
                            replace=False)
                else:
                    iou_sampled_inds = np.array(
                        iou_sampling_neg_inds, dtype=np.int)
                num_expected_floor = num_expected - len(iou_sampled_inds)
                if len(floor_neg_inds) > num_expected_floor:
                    sampled_floor_inds = np.random.choice(
                        floor_neg_inds, size=num_expected_floor, replace=False)
                else:
                    sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
                sampled_inds = np.concatenate(
                    (sampled_floor_inds, iou_sampled_inds))
                if len(sampled_inds) < num_expected:
                    num_extra = num_expected - len(sampled_inds)
                    extra_inds = np.array(list(neg_set - set(sampled_inds)))
                    if len(extra_inds) > num_extra:
                        extra_inds = np.random.choice(
                            extra_inds, size=num_extra, replace=False)
                    sampled_inds = np.concatenate((sampled_inds, extra_inds))
                return sampled_inds

        def _sample_rois(rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
                         batch_size_per_im, fg_fraction, fg_thresh,
                         bg_thresh_hi, bg_thresh_lo, bbox_reg_weights,
                         class_nums, use_random, is_cls_agnostic,
                         is_cascade_rcnn):
            rois_per_image = int(batch_size_per_im)
            fg_rois_per_im = int(np.round(fg_fraction * rois_per_image))

            # Roidb
            im_scale = im_info[2]
            inv_im_scale = 1. / im_scale
            rpn_rois = rpn_rois * inv_im_scale
            if is_cascade_rcnn:
                rpn_rois = rpn_rois[gt_boxes.shape[0]:, :]
            boxes = np.vstack([gt_boxes, rpn_rois])
            gt_overlaps = np.zeros((boxes.shape[0], class_nums))
            box_to_gt_ind_map = np.zeros((boxes.shape[0]), dtype=np.int32)
            if len(gt_boxes) > 0:
                proposal_to_gt_overlaps = bbox_overlaps(boxes, gt_boxes)

                overlaps_argmax = proposal_to_gt_overlaps.argmax(axis=1)
                overlaps_max = proposal_to_gt_overlaps.max(axis=1)
                # Boxes which with non-zero overlap with gt boxes
                overlapped_boxes_ind = np.where(overlaps_max > 0)[0]

                overlapped_boxes_gt_classes = gt_classes[overlaps_argmax[
                    overlapped_boxes_ind]]

                for idx in range(len(overlapped_boxes_ind)):
                    gt_overlaps[overlapped_boxes_ind[
                        idx], overlapped_boxes_gt_classes[idx]] = overlaps_max[
                            overlapped_boxes_ind[idx]]
                    box_to_gt_ind_map[overlapped_boxes_ind[
                        idx]] = overlaps_argmax[overlapped_boxes_ind[idx]]

            crowd_ind = np.where(is_crowd)[0]
            gt_overlaps[crowd_ind] = -1

            max_overlaps = gt_overlaps.max(axis=1)
            max_classes = gt_overlaps.argmax(axis=1)

            # Cascade RCNN Decode Filter
            if is_cascade_rcnn:
                ws = boxes[:, 2] - boxes[:, 0] + 1
                hs = boxes[:, 3] - boxes[:, 1] + 1
                keep = np.where((ws > 0) & (hs > 0))[0]
                boxes = boxes[keep]
                max_overlaps = max_overlaps[keep]
                fg_inds = np.where(max_overlaps >= fg_thresh)[0]
                bg_inds = np.where((max_overlaps < bg_thresh_hi) & (
                    max_overlaps >= bg_thresh_lo))[0]
                fg_rois_per_this_image = fg_inds.shape[0]
                bg_rois_per_this_image = bg_inds.shape[0]
            else:
                # Foreground
                fg_inds = np.where(max_overlaps >= fg_thresh)[0]
                fg_rois_per_this_image = np.minimum(fg_rois_per_im,
                                                    fg_inds.shape[0])
                # Sample foreground if there are too many
                if fg_inds.shape[0] > fg_rois_per_this_image:
                    if use_random:
                        fg_inds = _sample_pos(max_overlaps, max_classes,
                                              fg_inds, fg_rois_per_this_image)
                fg_inds = fg_inds[:fg_rois_per_this_image]

                # Background
                bg_inds = np.where((max_overlaps < bg_thresh_hi) & (
                    max_overlaps >= bg_thresh_lo))[0]
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
                bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                                    bg_inds.shape[0])
                assert bg_rois_per_this_image >= 0, "bg_rois_per_this_image must be >= 0 but got {}".format(
                    bg_rois_per_this_image)

                # Sample background if there are too many
                if bg_inds.shape[0] > bg_rois_per_this_image:
                    if use_random:
                        # libra neg sample
                        bg_inds = _sample_neg(
                            max_overlaps,
                            max_classes,
                            bg_inds,
                            bg_rois_per_this_image,
                            num_bins=num_bins,
                            bg_thresh_hi=bg_thresh_hi)
                bg_inds = bg_inds[:bg_rois_per_this_image]

            keep_inds = np.append(fg_inds, bg_inds)
            sampled_labels = max_classes[keep_inds]  # N x 1
            sampled_labels[fg_rois_per_this_image:] = 0
            sampled_boxes = boxes[keep_inds]  # N x 324
            sampled_gts = gt_boxes[box_to_gt_ind_map[keep_inds]]
            sampled_gts[fg_rois_per_this_image:, :] = gt_boxes[0]
            bbox_label_targets = _compute_targets(
                sampled_boxes, sampled_gts, sampled_labels, bbox_reg_weights)
            bbox_targets, bbox_inside_weights = _expand_bbox_targets(
                bbox_label_targets, class_nums, is_cls_agnostic)
            bbox_outside_weights = np.array(
                bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)
            # Scale rois
            sampled_rois = sampled_boxes * im_scale

            # Faster RCNN blobs
            frcn_blobs = dict(
                rois=sampled_rois,
                labels_int32=sampled_labels,
                bbox_targets=bbox_targets,
                bbox_inside_weights=bbox_inside_weights,
                bbox_outside_weights=bbox_outside_weights)
            return frcn_blobs

        def _compute_targets(roi_boxes, gt_boxes, labels, bbox_reg_weights):
            assert roi_boxes.shape[0] == gt_boxes.shape[0]
            assert roi_boxes.shape[1] == 4
            assert gt_boxes.shape[1] == 4

            targets = np.zeros(roi_boxes.shape)
            bbox_reg_weights = np.asarray(bbox_reg_weights)
            targets = box_to_delta(
                ex_boxes=roi_boxes,
                gt_boxes=gt_boxes,
                weights=bbox_reg_weights)

            return np.hstack([labels[:, np.newaxis], targets]).astype(
                np.float32, copy=False)

        def _expand_bbox_targets(bbox_targets_input, class_nums,
                                 is_cls_agnostic):
            class_labels = bbox_targets_input[:, 0]
            fg_inds = np.where(class_labels > 0)[0]
            bbox_targets = np.zeros((class_labels.shape[0], 4 * class_nums
                                     if not is_cls_agnostic else 4 * 2))
            bbox_inside_weights = np.zeros(bbox_targets.shape)
            for ind in fg_inds:
                class_label = int(class_labels[
                    ind]) if not is_cls_agnostic else 1
                start_ind = class_label * 4
                end_ind = class_label * 4 + 4
                bbox_targets[ind, start_ind:end_ind] = bbox_targets_input[ind,
                                                                          1:]
                bbox_inside_weights[ind, start_ind:end_ind] = (1.0, 1.0, 1.0,
                                                               1.0)
            return bbox_targets, bbox_inside_weights

        def generate_func(
                rpn_rois,
                gt_classes,
                is_crowd,
                gt_boxes,
                im_info, ):
            rpn_rois_lod = rpn_rois.lod()[0]
            gt_classes_lod = gt_classes.lod()[0]

            # convert
            rpn_rois = np.array(rpn_rois)
            gt_classes = np.array(gt_classes)
            is_crowd = np.array(is_crowd)
            gt_boxes = np.array(gt_boxes)
            im_info = np.array(im_info)

            rois = []
            labels_int32 = []
            bbox_targets = []
            bbox_inside_weights = []
            bbox_outside_weights = []
            lod = [0]

            for idx in range(len(rpn_rois_lod) - 1):
                rois_si = rpn_rois_lod[idx]
                rois_ei = rpn_rois_lod[idx + 1]

                gt_si = gt_classes_lod[idx]
                gt_ei = gt_classes_lod[idx + 1]
                frcn_blobs = _sample_rois(
                    rpn_rois[rois_si:rois_ei], gt_classes[gt_si:gt_ei],
                    is_crowd[gt_si:gt_ei], gt_boxes[gt_si:gt_ei], im_info[idx],
                    batch_size_per_im, fg_fraction, fg_thresh, bg_thresh_hi,
                    bg_thresh_lo, bbox_reg_weights, class_nums, use_random,
                    is_cls_agnostic, is_cascade_rcnn)
                lod.append(frcn_blobs['rois'].shape[0] + lod[-1])
                rois.append(frcn_blobs['rois'])
                labels_int32.append(frcn_blobs['labels_int32'].reshape(-1, 1))
                bbox_targets.append(frcn_blobs['bbox_targets'])
                bbox_inside_weights.append(frcn_blobs['bbox_inside_weights'])
                bbox_outside_weights.append(frcn_blobs['bbox_outside_weights'])

            rois = np.vstack(rois)
            labels_int32 = np.vstack(labels_int32)
            bbox_targets = np.vstack(bbox_targets)
            bbox_inside_weights = np.vstack(bbox_inside_weights)
            bbox_outside_weights = np.vstack(bbox_outside_weights)

            # create lod-tensor for return
            # notice that the func create_lod_tensor does not work well here
            ret_rois = fluid.LoDTensor()
            ret_rois.set_lod([lod])
            ret_rois.set(rois.astype("float32"), fluid.CPUPlace())

            ret_labels_int32 = fluid.LoDTensor()
            ret_labels_int32.set_lod([lod])
            ret_labels_int32.set(
                labels_int32.astype("int32"), fluid.CPUPlace())

            ret_bbox_targets = fluid.LoDTensor()
            ret_bbox_targets.set_lod([lod])
            ret_bbox_targets.set(
                bbox_targets.astype("float32"), fluid.CPUPlace())

            ret_bbox_inside_weights = fluid.LoDTensor()
            ret_bbox_inside_weights.set_lod([lod])
            ret_bbox_inside_weights.set(
                bbox_inside_weights.astype("float32"), fluid.CPUPlace())

            ret_bbox_outside_weights = fluid.LoDTensor()
            ret_bbox_outside_weights.set_lod([lod])
            ret_bbox_outside_weights.set(
                bbox_outside_weights.astype("float32"), fluid.CPUPlace())

            return ret_rois, ret_labels_int32, ret_bbox_targets, ret_bbox_inside_weights, ret_bbox_outside_weights

        rois = create_tmp_var(
            fluid.default_main_program(),
            name=None,  #'rois',
            dtype='float32',
            shape=[-1, 4], )
        bbox_inside_weights = create_tmp_var(
            fluid.default_main_program(),
            name=None,  #'bbox_inside_weights',
            dtype='float32',
            shape=[-1, 8 if self.is_cls_agnostic else self.class_nums * 4], )
        bbox_outside_weights = create_tmp_var(
            fluid.default_main_program(),
            name=None,  #'bbox_outside_weights',
            dtype='float32',
            shape=[-1, 8 if self.is_cls_agnostic else self.class_nums * 4], )
        bbox_targets = create_tmp_var(
            fluid.default_main_program(),
            name=None,  #'bbox_targets',
            dtype='float32',
            shape=[-1, 8 if self.is_cls_agnostic else self.class_nums * 4], )
        labels_int32 = create_tmp_var(
            fluid.default_main_program(),
            name=None,  #'labels_int32',
            dtype='int32',
            shape=[-1, 1], )

        outs = [
            rois, labels_int32, bbox_targets, bbox_inside_weights,
            bbox_outside_weights
        ]

        fluid.layers.py_func(
            func=generate_func,
            x=[rpn_rois, gt_classes, is_crowd, gt_boxes, im_info],
            out=outs)
        return outs


class BBoxAssigner(object):
    def __init__(self,
                 batch_size_per_im=512,
                 fg_fraction=.25,
                 fg_thresh=.5,
                 bg_thresh_hi=.5,
                 bg_thresh_lo=0.,
                 bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                 num_classes=81,
                 shuffle_before_sample=True):
        super(BBoxAssigner, self).__init__()
        self.batch_size_per_im = batch_size_per_im
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh_hi = bg_thresh_hi
        self.bg_thresh_lo = bg_thresh_lo
        self.bbox_reg_weights = bbox_reg_weights
        self.class_nums = num_classes
        self.use_random = shuffle_before_sample

    def __call__(self, rpn_rois, gt_classes, is_crowd, gt_boxes, im_info):
        return fluid.layers.generate_proposal_labels(
            rpn_rois=rpn_rois,
            gt_classes=gt_classes,
            is_crowd=is_crowd,
            gt_boxes=gt_boxes,
            im_info=im_info,
            batch_size_per_im=self.batch_size_per_im,
            fg_fraction=self.fg_fraction,
            fg_thresh=self.fg_thresh,
            bg_thresh_hi=self.bg_thresh_hi,
            bg_thresh_lo=self.bg_thresh_lo,
            bbox_reg_weights=self.bbox_reg_weights,
            class_nums=self.class_nums,
            use_random=self.use_random)
