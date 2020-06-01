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
from paddle import fluid


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