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

import numpy as np
import cv2
import copy
import random
import math
from PIL import Image
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from .functions import normalize, horizontal_flip, permute, vertical_flip, center_crop

interp_list = [
    cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
    cv2.INTER_LANCZOS4
]


class Transform:
    """分类Transform的基类
    """

    def __init__(self):
        pass

    def apply_im(self, im):
        pass

    def apply_mask(self, mask):
        pass

    def __call__(self, sample):
        im = sample['im']
        sample['im'] = self.apply_im(im)
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)

        return sample


class Compose(Transform):
    """根据数据预处理/增强算子对输入数据进行操作。
       所有操作的输入图像流形状均是[H, W, C]，其中H为图像高，W为图像宽，C为图像通道数。
    Args:
        transforms (list): 数据预处理/增强算子。
    Raises:
        TypeError: 形参数据类型不满足需求。
        ValueError: 数据长度不匹配。
    """

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

    def __call__(self, sample):
        if self.apply_im_only and 'mask' in sample:
            mask = sample['mask']
            mask_backup = copy.deepcopy(mask)
            del sample['mask']

        sample = self.decode_image(sample)

        for op in self.transforms:
            sample = op(sample)

        if self.arrange_outputs is not None:
            if self.apply_im_only:
                sample['mask'] = mask_backup
            outputs = self.arrange_outputs(sample)

        return outputs


class Decode(Transform):
    def __init__(self):
        self.to_rgb = True

    def read_img(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    def apply_im(self, im_path):
        if isinstance(im_path, str):
            try:
                im = self.read_img(im_path)
            except:
                raise ValueError('Cannot read the im file {}!'.format(im_path))
        else:
            im = im_path

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

    def __call__(self, sample):
        im = sample['im']
        sample['im'] = self.apply_im(im)
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)
            im_height, im_width, _ = sample['im'].shape
            se_height, se_width = sample['mask'].shape
            if im_height != se_height or im_width != se_width:
                raise Exception(
                    "The height or width of the im is not same as the mask")
        return sample


class Resize(Transform):
    def __init__(self, height, width, interp=cv2.INTER_LINEAR):
        self.target_h = height
        self.target_w = width
        self.interp = interp

    def apply_im(self, im, interp):
        im = cv2.resize(
            im, (self.target_w, self.target_h), interpolation=interp)
        return im

    def apply_mask(self, mask):
        mask = cv2.resize(
            mask, (self.target_w, self.target_h),
            interpolation=cv2.INTER_NEAREST)
        return mask

    def __call__(self, sample):
        interp = self.interp
        if self.interp == "RANDOM":
            interp = random.choice(interp_list)
        im = sample['im']
        sample['im'] = self.apply_im(im, interp)
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)

        return sample


class ResizeByShort(Transform):
    def __init__(self, short_size=256, max_size=-1, interp=cv2.INTER_LINEAR):
        self.short_size = short_size
        self.max_size = max_size
        self.interp = interp
        self.target_h = None
        self.target_w = None

    def apply_im(self, im, interp):
        im_short_size = min(im.shape[0], im.shape[1])
        im_long_size = max(im.shape[0], im.shape[1])
        scale = float(self.short_size) / im_short_size
        if 0 < self.max_size < np.round(scale * im_long_size):
            scale = float(self.max_size) / float(im_long_size)
        self.target_w = int(round(im.shape[1] * scale))
        self.target_h = int(round(im.shape[0] * scale))
        im = cv2.resize(
            im, (self.target_w, self.target_h), interpolation=interp)
        return im

    def __call__(self, sample):
        interp = self.interp
        if self.interp == "RANDOM":
            interp = random.choice(interp_list)
        im = sample['im']
        sample['im'] = self.apply_im(im, interp)
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)

        return sample


class RandomHorizontalFlip(Transform):
    def __init__(self, prob=0.5):
        self.prob = prob

    def apply_im(self, im):
        im = horizontal_flip(im)
        return im

    def apply_mask(self, mask):
        mask = horizontal_flip(mask)
        return mask

    def __call__(self, sample):
        if random.random() < self.prob:
            im = sample['im']
            sample['im'] = self.apply_im(im)
            if 'mask' in sample:
                mask = sample['mask']
                sample['mask'] = self.apply_mask(mask)
        return sample


class RandomVerticalFlip(Transform):
    """以一定的概率对图像进行随机垂直翻转，模型训练时的数据增强操作。
    Args:
        prob (float): 随机垂直翻转的概率。默认为0.5。
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def apply_im(self, im):
        im = vertical_flip(im)
        return im

    def apply_mask(self, mask):
        mask = vertical_flip(mask)
        return mask

    def __call__(self, sample):
        if random.random() < self.prob:
            im = sample['im']
            sample['im'] = self.apply_im(im)
            if 'mask' in sample:
                mask = sample['mask']
                sample['mask'] = self.apply_mask(mask)
        return sample


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
                'Std should not have 0, but received is {}'.format(std))
        if is_scale:
            if reduce(lambda x, y: x * y,
                      [a - b for a, b in zip(max_val, min_val)]) == 0:
                raise ValueError(
                    '(max_val - min_val) should not have 0, but received is {}'.
                    format(max_val - min_val))

        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val
        self.is_scale = is_scale

    def apply_im(self, im):
        im = im.astype(np.float32)
        mean = np.asarray(
            self.mean, dtype=np.float32)[np.newaxis, np.newaxis, :]
        std = np.asarray(self.std, dtype=np.float32)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std, self.min_val, self.max_val)
        return im

    def __call__(self, sample):
        im = sample['im']
        sample['im'] = self.apply_im(im)

        return sample


class CenterCrop(Transform):
    """以图像中心点扩散裁剪长宽为`crop_size`的正方形
    1. 计算剪裁的起始点。
    2. 剪裁图像。
    Args:
        crop_size (int): 裁剪的目标边长。默认为224。
    """

    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def apply_im(self, im):
        im = center_crop(im, self.crop_size)

        return im

    def apply_mask(self, mask):
        mask = center_crop(mask)
        return mask

    def __call__(self, sample):
        im = sample['im']
        sample['im'] = self.apply_im(im)
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)
        return sample


class RandomCrop(Transform):
    """对图像进行随机剪裁，模型训练时的数据增强操作。
    1. 根据lower_scale、lower_ratio、upper_ratio计算随机剪裁的高、宽。
    2. 根据随机剪裁的高、宽随机选取剪裁的起始点。
    3. 剪裁图像。
    4. 调整剪裁后的图像的大小到crop_size*crop_size。
    Args:
        crop_size (int): 随机裁剪后重新调整的目标边长。默认为224。
        lower_scale (float): 裁剪面积相对原面积比例的最小限制。默认为0.08。
        lower_ratio (float): 宽变换比例的最小限制。默认为3. / 4。
        upper_ratio (float): 宽变换比例的最大限制。默认为4. / 3。
    """

    def __init__(self,
                 crop_size=224,
                 lower_scale=0.08,
                 lower_ratio=3. / 4,
                 upper_ratio=4. / 3):
        self.crop_size = crop_size
        self.lower_scale = lower_scale
        self.lower_ratio = lower_ratio
        self.upper_ratio = upper_ratio
        self.w, self.h, self.i, self.j = None, None, None, None

    def generate_crop_info(self,
                           im,
                           lower_scale=0.08,
                           lower_ratio=3. / 4,
                           upper_ratio=4. / 3):
        scale = [lower_scale, 1.0]
        ratio = [lower_ratio, upper_ratio]
        aspect_ratio = math.sqrt(np.random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio
        bound = min((float(im.shape[0]) / im.shape[1]) / (h**2),
                    (float(im.shape[1]) / im.shape[0]) / (w**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)
        target_area = im.shape[0] * im.shape[1] * np.random.uniform(scale_min,
                                                                    scale_max)
        target_size = math.sqrt(target_area)
        self.w = int(target_size * w)
        self.h = int(target_size * h)
        self.i = np.random.randint(0, im.shape[0] - h + 1)
        self.j = np.random.randint(0, im.shape[1] - w + 1)

    def apply_im(self, im):
        im = im[self.i:self.i + self.h, self.j:self.j + self.w, :]
        im = cv2.resize(im, (self.crop_size, self.crop_size))
        return im

    def apply_mask(self, mask):
        mask = mask[self.i:self.i + self.h, self.j:self.j + self.w, ...]
        mask = cv2.resize(mask, (self.crop_size, self.crop_size))
        return mask

    def __call__(self, sample):
        im = sample['im']
        self.generate_crop_info(im, self.lower_scale, self.lower_ratio,
                                self.upper_ratio)
        sample['im'] = self.apply_im(im)

        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)
        return sample


class Padding(Transform):
    def __init__(self,
                 height,
                 width,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        if isinstance(target_size, (list, tuple)):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        else:
            raise TypeError(
                "Type of target_size is invalid. It should be list or tuple, now is {}"
                .format(type(target_size)))
        self.target_h = height
        self.target_w = width
        self.target_size = target_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply_im(self, im):
        im_height, im_width = im.shape[0:2]
        pad_height = self.target_h - im_height
        pad_width = self.target_w - im_width
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                "The size of image should be less than 'target_size', "
                "but the size of image ({}, {}) is larger than `target_size` ({}, {})"
                .format(im_width, im_height, self.target_w, self.target_h))
        im = cv2.copyMakeBorder(
            im,
            0,
            pad_height,
            0,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=self.im_padding_value)
        return im

    def apply_mask(self, mask):
        mask_height, mask_width = mask.shape[0:2]
        pad_height = self.target_h - mask_height
        pad_width = self.target_w - mask_width
        mask = cv2.copyMakeBorder(
            mask,
            0,
            pad_height,
            0,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=self.label_padding_value)
        return mask

    def __call__(self, sample):
        im = sample['im']
        sample['im'] = self.apply_im(im)
        if 'mask' in sample:
            mask = sample['mask']
            sample['mask'] = self.apply_mask(mask)
        return sample


class ArrangeSegmenter(Transform):
    def __init__(self, mode):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, outputs):
        im, mask = outputs['im'], outputs['mask']

        im = permute(im, False)
        if self.mode == 'train':
            mask = mask.astype('int64')
            return im, mask
        if self.mode == 'eval':
            mask = np.asarray(Image.open(mask))
            mask = mask[np.newaxis, :, :].astype('int64')
            return im, mask
        if self.mode == 'test':
            return im,


class ArrangeClassifier(Transform):
    def __init__(self, mode):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, outputs):
        im = outputs['im']
        im = permute(im, False)
        return im,
