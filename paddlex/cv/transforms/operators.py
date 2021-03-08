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

import sys
import os.path as osp
import six
import numpy as np
import cv2
import copy
import imghdr
import random
from PIL import Image
from collections import OrderedDict
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from .functions import normalize, horizontal_flip, permute

interp_list = [
    cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
    cv2.INTER_LANCZOS4
]


class Transform:
    """ transform基类
    """

    def __init__(self):
        pass

    def apply_im(self, im):
        pass

    def apply_mask(self, mask):
        pass

    def __call(self, im, mask=None):
        self.apply_im(im)
        if mask is not None:
            self.apply_mask(mask)
        outputs = {
            'im': im,
            'mask': mask,
        }

        return outputs


class Compose(Transform):
    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError(
                'Type of transforms is invalid. Must be List, but recieved is {}'
                .format(type(transforms)))
        if len(transforms) < 1:
            raise ValueError(
                'Length of transforms must not be less than 1, but recieved is {}'
                .format(len(transforms)))
        self.transforms = transforms
        self.decode_image = Decode()
        self.arrange_outputs = None
        self.apply_im_only = False

    def __call__(self, im, mask=None):
        if self.apply_im_only:
            mask_backup = copy.deepcopy(mask)
            mask = None

        outputs = self.decode_image(im, mask)
        im = outputs['im']
        mask = outputs['mask']

        for op in self.transforms:
            outputs = op(im, mask)
            im = outputs['im']
            mask = outputs['mask']

        if self.arrange_outputs is not None:
            if self.apply_im_only:
                mask = mask_backup
            outputs = self.arrange_outputs(im, mask)
        return outputs


class Decode(Transform):
    def __init__(self):
        self.to_rgb = True

    def read_img(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    def apply_im(self, im_path):
        try:
            im = self.read_img(im_path)
        except:
            raise ValueError('Cannot read the im file {}!'.format(im_path))

        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im

    def apply_mask(self, mask):
        try:
            mask = np.asarray(Image.open(mask))
        except:
            raise ValueError("Cannot read the mask file {}!".format(mask))
        if len(mask.shape) != 2:
            raise Exception(
                "Mask should be a 1-channel image, but recevied is a {}-channel image.".
                format(mask.shape[2]))
        return mask

    def __call__(self, im, mask=None):
        im = self.apply_im(im)
        if mask is not None:
            mask = self.apply_mask(mask)
            im_height, im_width, _ = im.shape
            se_height, se_width = mask.shape
            if im_height != se_height or im_width != se_width:
                raise Exception(
                    "The height or width of the im is not same as the mask")

        outputs = {
            'im': im,
            'mask': mask,
        }

        return outputs


class Resize(Transform):
    def __init__(self, height, width, interp=cv2.INTER_LINEAR):
        self.height = height
        self.width = width
        self.interp = interp

    def apply_im(self, im, interp):
        im = cv2.resize(im, (self.width, self.height), interpolation=interp)
        return im

    def apply_mask(self, mask):
        mask = cv2.resize(
            mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return mask

    def __call__(self, im, mask=None):
        interp = self.interp
        if self.interp == "RANDOM":
            interp = random.choice(interp_list)
        im = self.apply_im(im, interp)
        if mask is not None:
            mask = self.apply_mask(mask)
        outputs = {
            'im': im,
            'mask': mask,
        }
        return outputs


class RandomHorizontalFlip(Transform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply_im(self, im):
        im = horizontal_flip(im)
        return im

    def apply_mask(self, mask):
        mask = horizontal_flip(mask)
        return mask

    def __call__(self, im, mask=None):
        if random.random() < self.prob:
            im = self.apply_im(im)
            if mask is not None:
                mask = self.apply_mask(mask)

        outputs = {
            'im': im,
            'mask': mask,
        }
        return outputs


class Normalize(Transform):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 min_val=[0, 0, 0],
                 max_val=[255., 255., 255.],
                 is_scale=True):
        from functools import reduce
        if reduce(lambda x, y: x * y, std) == 0:
            raise ValueError(
                'Std should not have 0, but recieved is {}'.format(std))
        if is_scale:
            if reduce(lambda x, y: x * y,
                      [a - b for a, b in zip(max_val, min_val)]) == 0:
                raise ValueError(
                    '(max_val - min_val) should not have 0, but recieved is {}'.
                    format(max_val - min_val))

        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.is_scale = is_scale

    def apply_im(self, im):
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std, self.min_val, self.max_val)
        return im

    def __call__(self, im, mask=None):
        im = self.apply_im(im)
        outputs = {
            'im': im,
            'mask': mask,
        }
        return outputs


class ArrangeSegmenter(Transform):
    def __init__(self, mode):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, im, mask=None):
        im = permute(im, False)
        if self.mode == 'train':
            mask = mask[np.newaxis, :, :].astype('int64')
            return (im, mask)
        if self.mode == 'eval':
            mask = np.asarray(Image.open(mask))
            mask = mask[np.newaxis, :, :].astype('int64')
            return (im, mask)
        if self.mode == 'test':
            return (im, )
