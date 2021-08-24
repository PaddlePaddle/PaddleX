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

import traceback
import random
import numpy as np
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from paddle.fluid.dataloader.collate import default_collate_fn
from .operators import Transform, Resize, ResizeByShort, _Permute, interp_dict
from .box_utils import jaccard_overlap
from paddlex.utils import logging


class BatchCompose(Transform):
    def __init__(self, batch_transforms=None, collate_batch=True):
        super(BatchCompose, self).__init__()
        self.batch_transforms = batch_transforms
        self.collate_batch = collate_batch

    def __call__(self, samples):
        if self.batch_transforms is not None:
            for op in self.batch_transforms:
                try:
                    samples = op(samples)
                except Exception as e:
                    stack_info = traceback.format_exc()
                    logging.warning("fail to map batch transform [{}] "
                                    "with error: {} and stack:\n{}".format(
                                        op, e, str(stack_info)))
                    raise e

        samples = _Permute()(samples)

        extra_key = ['h', 'w', 'flipped']
        for k in extra_key:
            for sample in samples:
                if k in sample:
                    sample.pop(k)

        if self.collate_batch:
            batch_data = default_collate_fn(samples)
        else:
            batch_data = {}
            for k in samples[0].keys():
                tmp_data = []
                for i in range(len(samples)):
                    tmp_data.append(samples[i][k])
                if not 'gt_' in k and not 'is_crowd' in k and not 'difficult' in k:
                    tmp_data = np.stack(tmp_data, axis=0)
                batch_data[k] = tmp_data
        return batch_data


class BatchRandomResize(Transform):
    """
    Resize a batch of input to random sizes.

    Attention：If interp is 'RANDOM', the interpolation method will be chose randomly.

    Args:
        target_sizes (List[int], List[list or tuple] or Tuple[list or tuple]):
            Multiple target sizes, each target size is an int or list/tuple of length 2.
        interp ({'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}, optional):
            Interpolation method of resize. Defaults to 'LINEAR'.
    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.

    See Also:
        RandomResize: Resize input to random sizes.
    """

    def __init__(self, target_sizes, interp='NEAREST'):
        super(BatchRandomResize, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("interp should be one of {}".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(target_sizes, list), \
            "target_size must be List"
        for i, item in enumerate(target_sizes):
            if isinstance(item, int):
                target_sizes[i] = (item, item)
        self.target_size = target_sizes

    def __call__(self, samples):
        height, width = random.choice(self.target_size)
        resizer = Resize((height, width), interp=self.interp)
        samples = resizer(samples)

        return samples


class BatchRandomResizeByShort(Transform):
    """Resize a batch of input to random sizes with keeping the aspect ratio.

    Attention：If interp is 'RANDOM', the interpolation method will be chose randomly.

    Args:
        short_sizes (List[int], Tuple[int]): Target sizes of the shorter side of the image(s).
        max_size (int, optional): The upper bound of longer side of the image(s).
            If max_size is -1, no upper bound is applied. Defaults to -1.
        interp ({'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}, optional):
            Interpolation method of resize. Defaults to 'LINEAR'.

    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.

    See Also:
        RandomResizeByShort: Resize input to random sizes with keeping the aspect ratio.
    """

    def __init__(self, short_sizes, max_size=-1, interp='NEAREST'):
        super(BatchRandomResizeByShort, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("interp should be one of {}".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(short_sizes, list), \
            "short_sizes must be List"

        self.short_sizes = short_sizes
        self.max_size = max_size

    def __call__(self, samples):
        short_size = random.choice(self.short_sizes)
        resizer = ResizeByShort(
            short_size=short_size, max_size=self.max_size, interp=self.interp)

        samples = resizer(samples)

        return samples


class _BatchPadding(Transform):
    def __init__(self, pad_to_stride=0):
        super(_BatchPadding, self).__init__()
        self.pad_to_stride = pad_to_stride

    def __call__(self, samples):
        coarsest_stride = self.pad_to_stride
        max_shape = np.array([data['image'].shape for data in samples]).max(
            axis=0)
        if coarsest_stride > 0:
            max_shape[0] = int(
                np.ceil(max_shape[0] / coarsest_stride) * coarsest_stride)
            max_shape[1] = int(
                np.ceil(max_shape[1] / coarsest_stride) * coarsest_stride)
        for data in samples:
            im = data['image']
            im_h, im_w, im_c = im.shape[:]
            padding_im = np.zeros(
                (max_shape[0], max_shape[1], im_c), dtype=np.float32)
            padding_im[:im_h, :im_w, :] = im
            data['image'] = padding_im

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
