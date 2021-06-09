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
"""
function:
    transforms for detection in PaddleX<2.0
"""

import numpy as np
from .operators import Transform, Compose, ResizeByShort, Resize, RandomHorizontalFlip, Normalize, MixupImage
from .operators import RandomExpand as dy_RandomExpand
from .operators import RandomCrop as dy_RandomCrop
from .functions import is_poly, expand_poly, expand_rle

__all__ = [
    'Compose', 'ResizeByShort', 'Resize', 'RandomHorizontalFlip', 'Normalize',
    'MixupImage', 'Padding', 'RandomExpand', 'RandomCrop'
]


class Padding(Transform):
    """1.将图像的长和宽padding至coarsest_stride的倍数。如输入图像为[300, 640],
       `coarest_stride`为32，则由于300不为32的倍数，因此在图像最右和最下使用0值
       进行padding，最终输出图像为[320, 640]。
       2.或者，将图像的长和宽padding到target_size指定的shape，如输入的图像为[300，640]，
         a. `target_size` = 960，在图像最右和最下使用0值进行padding，最终输出
            图像为[960, 960]。
         b. `target_size` = [640, 960]，在图像最右和最下使用0值进行padding，最终
            输出图像为[640, 960]。
    1. 如果coarsest_stride为1，target_size为None则直接返回。
    2. 获取图像的高H、宽W。
    3. 计算填充后图像的高H_new、宽W_new。
    4. 构建大小为(H_new, W_new, 3)像素值为0的np.ndarray，
       并将原图的np.ndarray粘贴于左上角。
    Args:
        coarsest_stride (int): 填充后的图像长、宽为该参数的倍数，默认为1。
        target_size (int|list|tuple): 填充后的图像长、宽，默认为None，coarset_stride优先级更高。
    Raises:
        TypeError: 形参`target_size`数据类型不满足需求。
        ValueError: 形参`target_size`为(list|tuple)时，长度不满足需求。
    """

    def __init__(self, coarsest_stride=1, target_size=None):
        if target_size is not None:
            if not isinstance(target_size, int):
                if not isinstance(target_size, tuple) and not isinstance(
                        target_size, list):
                    raise TypeError(
                        "Padding: Type of target_size must in (int|list|tuple)."
                    )
                elif len(target_size) != 2:
                    raise ValueError(
                        "Padding: Length of target_size must equal 2.")
        super(Padding, self).__init__()
        self.coarsest_stride = coarsest_stride
        self.target_size = target_size

    def apply_im(self, image, padding_im_h, padding_im_w):
        im_h, im_w, im_c = image.shape
        padding_im = np.zeros(
            (padding_im_h, padding_im_w, im_c), dtype=np.float32)
        padding_im[:im_h, :im_w, :] = image
        return padding_im

    def apply_bbox(self, bbox):
        return bbox

    def apply_segm(self, segms, im_h, im_w, padding_im_h, padding_im_w):
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [expand_poly(poly, 0, 0) for poly in segm])
            else:
                # RLE format
                expanded_segms.append(
                    expand_rle(segm, 0, 0, im_h, im_w, padding_im_h,
                               padding_im_w))
        return expanded_segms

    def apply(self, sample):
        im_h, im_w, im_c = sample['image'].shape[:]

        if isinstance(self.target_size, int):
            padding_im_h = self.target_size
            padding_im_w = self.target_size
        elif isinstance(self.target_size, list) or isinstance(self.target_size,
                                                              tuple):
            padding_im_w = self.target_size[0]
            padding_im_h = self.target_size[1]
        elif self.coarsest_stride > 0:
            padding_im_h = int(
                np.ceil(im_h / self.coarsest_stride) * self.coarsest_stride)
            padding_im_w = int(
                np.ceil(im_w / self.coarsest_stride) * self.coarsest_stride)
        else:
            raise ValueError(
                "coarsest_stridei(>1) or target_size(list|int) need setting in Padding transform"
            )
        pad_height = padding_im_h - im_h
        pad_width = padding_im_w - im_w
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                'the size of image should be less than target_size, but the size of image ({}, {}), is larger than target_size ({}, {})'
                .format(im_w, im_h, padding_im_w, padding_im_h))
        sample['image'] = self.apply_im(sample['image'], padding_im_h,
                                        padding_im_w)
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'])
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h, im_w,
                                                padding_im_h, padding_im_w)

        return sample


class RandomExpand(dy_RandomExpand):
    def __init__(self,
                 ratio=4.,
                 prob=0.5,
                 fill_value=[123.675, 116.28, 103.53]):
        super(RandomExpand, self).__init__(
            upper_ratio=ratio, prob=prob, im_padding_value=fill_value)


class RandomCrop(dy_RandomCrop):
    def __init__(self,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__(
            crop_size=None,
            aspect_ratio=aspect_ratio,
            thresholds=thresholds,
            scaling=scaling,
            num_attempts=num_attempts,
            allow_no_crop=allow_no_crop,
            cover_all_box=cover_all_box)
