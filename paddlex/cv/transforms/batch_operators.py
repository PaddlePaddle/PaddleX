# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import multiprocessing as mp
import random
import numpy as np
import cv2
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from .operators import Transform, Resize, _Permute
from .box_utils import jaccard_overlap

MAIN_PID = os.getpid()


class BatchCompose(Transform):
    def __init__(self, batch_transforms):
        super(BatchCompose, self).__init__()
        if not isinstance(batch_transforms, list):
            raise TypeError(
                'Type of transforms is invalid. Must be List, but received is {}'
                .format(type(batch_transforms)))
        self.output_fields = mp.Manager().list([])
        self.batch_transforms = batch_transforms
        self.lock = mp.Lock()

    def __call__(self, samples):
        for op in self.batch_transforms:
            samples = op(samples)

        global MAIN_PID
        if os.getpid() == MAIN_PID and \
                isinstance(self.output_fields, mp.managers.ListProxy):
            self.output_fields = []

        if len(self.output_fields) == 0:
            self.lock.acquire()
            if len(self.output_fields) == 0:
                for k, v in samples[0].items():
                    self.output_fields.append(k)
            self.lock.release()
        samples = [[samples[i][k] for k in self.output_fields]
                   for i in range(len(samples))]
        samples = list(zip(*samples))
        samples = [np.stack(d, axis=0) for d in samples]

        return samples


class BatchRandomResize(Transform):
    """
    Resize image to target size randomly. random target_size and interpolation method
    Args:
        target_size (list): image target size, must be list of (int or list)
        keep_ratio (bool): whether keep_raio or not, default true
        interp (int): the interpolation method
    """

    def __init__(self, target_size, keep_ratio, interp=cv2.INTER_NEAREST):
        super(BatchRandomResize, self).__init__()
        self.keep_ratio = keep_ratio
        self.interp = interp
        assert isinstance(target_size, list), \
            "target_size must be List"
        for i, item in enumerate(target_size):
            if isinstance(item, int):
                target_size[i] = (item, item)
        self.target_size = target_size

    def __call__(self, samples):
        height, width = random.choice(self.target_size)

        resizer = Resize(
            height=height,
            width=width,
            keep_ratio=self.keep_ratio,
            interp=self.interp)
        samples = resizer(samples)

        return samples


class _Gt2YoloTarget(Transform):
    """
    Generate YOLOv3 targets by groud truth data, this operator is only used in
    fine grained YOLOv3 loss mode
    """

    def __init__(self,
                 anchors,
                 anchor_masks,
                 downsample_ratios,
                 num_classes=80,
                 iou_thresh=1.):
        super(_Gt2YoloTarget, self).__init__()
        self.anchors = anchors
        self.anchor_masks = anchor_masks
        self.downsample_ratios = downsample_ratios
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh

    def __call__(self, samples, context=None):
        assert len(self.anchor_masks) == len(self.downsample_ratios), \
            "anchor_masks', and 'downsample_ratios' should have same length."

        h, w = samples[0]['image'].shape[:2]
        an_hw = np.array(self.anchors) / np.array([[w, h]])
        for sample in samples:
            im = sample['image']
            gt_bbox = sample['gt_bbox']
            gt_class = sample['gt_class']
            if 'gt_score' not in sample:
                sample['gt_score'] = np.ones(
                    (gt_bbox.shape[0], 1), dtype=np.float32)
            gt_score = sample['gt_score']
            for i, (
                    mask, downsample_ratio
            ) in enumerate(zip(self.anchor_masks, self.downsample_ratios)):
                grid_h = int(h / downsample_ratio)
                grid_w = int(w / downsample_ratio)
                target = np.zeros(
                    (len(mask), 6 + self.num_classes, grid_h, grid_w),
                    dtype=np.float32)
                for b in range(gt_bbox.shape[0]):
                    gx, gy, gw, gh = gt_bbox[b, :]
                    cls = gt_class[b]
                    score = gt_score[b]
                    if gw <= 0. or gh <= 0. or score <= 0.:
                        continue

                    # find best match anchor index
                    best_iou = 0.
                    best_idx = -1
                    for an_idx in range(an_hw.shape[0]):
                        iou = jaccard_overlap(
                            [0., 0., gw, gh],
                            [0., 0., an_hw[an_idx, 0], an_hw[an_idx, 1]])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = an_idx

                    gi = int(gx * grid_w)
                    gj = int(gy * grid_h)

                    # gtbox should be regresed in this layes if best match
                    # anchor index in anchor mask of this layer
                    if best_idx in mask:
                        best_n = mask.index(best_idx)

                        # x, y, w, h, scale
                        target[best_n, 0, gj, gi] = gx * grid_w - gi
                        target[best_n, 1, gj, gi] = gy * grid_h - gj
                        target[best_n, 2, gj, gi] = np.log(
                            gw * w / self.anchors[best_idx][0])
                        target[best_n, 3, gj, gi] = np.log(
                            gh * h / self.anchors[best_idx][1])
                        target[best_n, 4, gj, gi] = 2.0 - gw * gh

                        # objectness record gt_score
                        target[best_n, 5, gj, gi] = score

                        # classification
                        target[best_n, 6 + cls, gj, gi] = 1.

                    # For non-matched anchors, calculate the target if the iou
                    # between anchor and gt is larger than iou_thresh
                    if self.iou_thresh < 1:
                        for idx, mask_i in enumerate(mask):
                            if mask_i == best_idx: continue
                            iou = jaccard_overlap(
                                [0., 0., gw, gh],
                                [0., 0., an_hw[mask_i, 0], an_hw[mask_i, 1]])
                            if iou > self.iou_thresh and target[idx, 5, gj,
                                                                gi] == 0.:
                                # x, y, w, h, scale
                                target[idx, 0, gj, gi] = gx * grid_w - gi
                                target[idx, 1, gj, gi] = gy * grid_h - gj
                                target[idx, 2, gj, gi] = np.log(
                                    gw * w / self.anchors[mask_i][0])
                                target[idx, 3, gj, gi] = np.log(
                                    gh * h / self.anchors[mask_i][1])
                                target[idx, 4, gj, gi] = 2.0 - gw * gh

                                # objectness record gt_score
                                target[idx, 5, gj, gi] = score

                                # classification
                                target[idx, 5 + cls, gj, gi] = 1.
                sample['target{}'.format(i)] = target

            # remove useless gt_class and gt_score after target calculated
            sample.pop('gt_class')
            sample.pop('gt_score')

        return samples
