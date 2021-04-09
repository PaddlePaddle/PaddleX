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

    def apply_bbox(self, bbox):
        pass

    def __call__(self, sample):
        sample['im'] = self.apply_im(sample['im'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'])

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
        super(Compose, self).__init__()
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
            mask_backup = copy.deepcopy(sample['mask'])
            del sample['mask']

        sample = self.decode_image(sample)

        for op in self.transforms:
            sample = op(sample)

        if self.arrange_outputs is not None:
            if self.apply_im_only:
                sample['mask'] = mask_backup
            sample = self.arrange_outputs(sample)

        return sample


class Decode(Transform):
    def __init__(self):
        super(Decode, self).__init__()
        self.to_rgb = True

    def read_img(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

    def apply_im(self, im_path):
        if isinstance(im_path, str):
            try:
                im = self.read_img(im_path)
            except:
                raise ValueError('Cannot read the image file {}!'.format(
                    im_path))
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
        sample['im'] = self.apply_im(sample['im'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
            im_height, im_width, _ = sample['im'].shape
            se_height, se_width = sample['mask'].shape
            if im_height != se_height or im_width != se_width:
                raise Exception(
                    "The height or width of the im is not same as the mask")
        return sample


class Resize(Transform):
    def __init__(self, height, width, interp=cv2.INTER_LINEAR):
        super(Resize, self).__init__()
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

    def apply_bbox(self, bbox, scale):
        im_scale_x, im_scale_y = scale
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, self.target_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, self.target_h)
        return bbox

    def __call__(self, sample):
        interp = self.interp
        if self.interp == "RANDOM":
            interp = random.choice(interp_list)
        im_h, im_w = sample['im'].shape[:2]
        sample['im'] = self.apply_im(sample['im'], interp)
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            im_scale_y = self.target_h / im_h
            im_scale_x = self.target_w / im_w
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y])

        return sample


class ResizeByShort(Transform):
    def __init__(self, short_size=256, max_size=-1, interp=cv2.INTER_LINEAR):
        super(ResizeByShort, self).__init__()
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

    def apply_mask(self, mask):
        mask = cv2.resize(
            mask, (self.target_w, self.target_h),
            interpolation=cv2.INTER_NEAREST)
        return mask

    def apply_bbox(self, bbox, scale):
        im_scale_x, im_scale_y = scale
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, self.target_w)
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, self.target_h)
        return bbox

    def __call__(self, sample):
        interp = self.interp
        if self.interp == "RANDOM":
            interp = random.choice(interp_list)
        im_h, im_w = sample['im'].shape[:2]
        sample['im'] = self.apply_im(sample['im'], interp)
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            im_scale_y = self.target_h / im_h
            im_scale_x = self.target_w / im_w
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'],
                                                [im_scale_x, im_scale_y])

        return sample


class RandomHorizontalFlip(Transform):
    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, im):
        im = horizontal_flip(im)
        return im

    def apply_mask(self, mask):
        mask = horizontal_flip(mask)
        return mask

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def __call__(self, sample):
        if random.random() < self.prob:
            _, im_w = sample['im'].shape[:2]
            sample['im'] = self.apply_im(sample['im'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_w)
        return sample


class RandomVerticalFlip(Transform):
    """以一定的概率对图像进行随机垂直翻转，模型训练时的数据增强操作。
    Args:
        prob (float): 随机垂直翻转的概率。默认为0.5。
    """

    def __init__(self, prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, im):
        im = vertical_flip(im)
        return im

    def apply_mask(self, mask):
        mask = vertical_flip(mask)
        return mask

    def apply_bbox(self, bbox, height):
        oldy1 = bbox[:, 1].copy()
        oldy2 = bbox[:, 3].copy()
        bbox[:, 0] = height - oldy2
        bbox[:, 2] = height - oldy1
        return bbox

    def __call__(self, sample):
        if random.random() < self.prob:
            im_h, _ = sample['im'].shape[:2]
            sample['im'] = self.apply_im(sample['im'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_h)
        return sample


class Normalize(Transform):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 min_val=[0, 0, 0],
                 max_val=[255., 255., 255.],
                 is_scale=True):
        super(Normalize, self).__init__()
        from functools import reduce
        if reduce(lambda x, y: x * y, std) == 0:
            raise ValueError(
                'Std should not have 0, but received is {}'.format(std))
        if is_scale:
            if reduce(lambda x, y: x * y,
                      [a - b for a, b in zip(max_val, min_val)]) == 0:
                raise ValueError(
                    '(max_val - min_val) should not have 0, but received is {}'.
                    format((np.asarray(max_val) - np.asarray(min_val)).tolist(
                    )))

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
        sample['im'] = self.apply_im(sample['im'])

        return sample


class CenterCrop(Transform):
    """以图像中心点扩散裁剪长宽为`crop_size`的正方形
    1. 计算剪裁的起始点。
    2. 剪裁图像。
    Args:
        crop_size (int): 裁剪的目标边长。默认为224。
    """

    def __init__(self, crop_size=224):
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def apply_im(self, im):
        im = center_crop(im, self.crop_size)

        return im

    def apply_mask(self, mask):
        mask = center_crop(mask)
        return mask

    def __call__(self, sample):
        sample['im'] = self.apply_im(sample['im'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
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
                 crop_size=None,
                 aspect_ratio=[.5, 2.],
                 thresholds=[.0, .1, .3, .5, .7, .9],
                 scaling=[.3, 1.],
                 num_attempts=50,
                 allow_no_crop=True,
                 cover_all_box=False):
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.thresholds = thresholds
        self.scaling = scaling
        self.num_attempts = num_attempts
        self.allow_no_crop = allow_no_crop
        self.cover_all_box = cover_all_box
        self.crop_box = None
        self.cropped_box = None
        self.valid_ids = None
        self.found = False

    def _generate_crop_info(self, sample):
        im_h, im_w = sample['im'].shape[:2]
        gt_bbox = sample['gt_bbox']
        thresholds = self.thresholds
        if thresholds is not None:
            if self.allow_no_crop:
                thresholds.append('no_crop')
            np.random.shuffle(thresholds)
            for thresh in thresholds:
                if thresh == 'no_crop':
                    break
                for i in range(self.num_attempts):
                    crop_box = self._get_crop_box(im_h, im_w)
                    if crop_box is None:
                        continue
                    self.crop_box = crop_box
                    iou = self._iou_matrix(
                        gt_bbox, np.array(
                            [self.crop_box], dtype=np.float32))
                    if iou.max() < thresh:
                        continue
                    if self.cover_all_box and iou.min() < thresh:
                        continue
                    self.cropped_box, self.valid_ids = self._crop_box_with_center_constraint(
                        gt_bbox, np.array(
                            self.crop_box, dtype=np.float32))
                    if self.valid_ids.size > 0:
                        self.found = True
                        break
        else:
            for i in range(self.num_attempts):
                crop_box = self._get_crop_box(im_h, im_w)
                if crop_box is None:
                    continue
                self.crop_box = crop_box
                self.found = True

    def _get_crop_box(self, im_h, im_w):
        scale = np.random.uniform(*self.scaling)
        if self.aspect_ratio is not None:
            min_ar, max_ar = self.aspect_ratio
            aspect_ratio = np.random.uniform(
                max(min_ar, scale**2), min(max_ar, scale**-2))
            h_scale = scale / np.sqrt(aspect_ratio)
            w_scale = scale * np.sqrt(aspect_ratio)
        else:
            h_scale = np.random.uniform(*self.scaling)
            w_scale = np.random.uniform(*self.scaling)
        crop_h = im_h * h_scale
        crop_w = im_w * w_scale
        if self.aspect_ratio is None:
            if crop_h / crop_w < 0.5 or crop_h / crop_w > 2.0:
                return None
        crop_h = int(crop_h)
        crop_w = int(crop_w)
        crop_y = np.random.randint(0, im_h - crop_h)
        crop_x = np.random.randint(0, im_w - crop_w)
        return [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]

    def _iou_matrix(self, a, b):
        tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
        br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

        area_i = np.prod(br_i - tl_i, axis=2) * (tl_i < br_i).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        area_o = (area_a[:, np.newaxis] + area_b - area_i)
        return area_i / (area_o + 1e-10)

    def _crop_box_with_center_constraint(self, box, crop):
        cropped_box = box.copy()

        cropped_box[:, :2] = np.maximum(box[:, :2], crop[:2])
        cropped_box[:, 2:] = np.minimum(box[:, 2:], crop[2:])
        cropped_box[:, :2] -= crop[:2]
        cropped_box[:, 2:] -= crop[:2]

        centers = (box[:, :2] + box[:, 2:]) / 2
        valid = np.logical_and(crop[:2] <= centers,
                               centers < crop[2:]).all(axis=1)
        valid = np.logical_and(
            valid, (cropped_box[:, :2] < cropped_box[:, 2:]).all(axis=1))

        return cropped_box, np.where(valid)[0]

    def apply_im(self, im, crop):
        x1, y1, x2, y2 = crop
        return im[y1:y2, x1:x2, :]

    def apply_mask(self, mask, crop):
        x1, y1, x2, y2 = crop
        return mask[y1:y2, x1:x2, :]

    def __call__(self, sample):
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            self._generate_crop_info(sample)
            if self.found:
                sample['im'] = self.apply_im(sample['image'], self.crop_box)
                sample['gt_bbox'] = np.take(
                    self.cropped_box, self.valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], self.valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], self.valid_ids, axis=0)
                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], self.valid_ids, axis=0)
        else:
            self.thresholds = None
            self._generate_crop_info(sample)
            if self.found:
                sample['im'] = self.apply_im(sample['image'], self.crop_box)
                if 'mask' in sample:
                    sample['mask'] = self.apply_mask(sample['mask'])

        if self.crop_size is not None:
            sample = Resize(self.crop_size, self.crop_size)(sample)

        return sample


class Padding(Transform):
    def __init__(self,
                 height,
                 width,
                 target_size,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        super(Padding, self).__init__()
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
        sample['im'] = self.apply_im(sample['im'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        return sample


class MixupImage(Transform):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha should be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta should be positive in {}".format(self))

    def apply_im(self, im1, im2, factor):
        h = max(im1.shape[0], im2.shape[0])
        w = max(im1.shape[1], im2.shape[1])
        img = np.zeros((h, w, im1.shape[2]), 'float32')
        img[:im1.shape[0], :im1.shape[1], :] = \
            im1.astype('float32') * factor
        img[:im2.shape[0], :im2.shape[1], :] += \
            im2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample):
        if not isinstance(sample, Sequence):
            return sample

        assert len(sample) == 2, 'mixup need two samples'

        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            return sample[0]
        if factor <= 0.0:
            return sample[1]
        im = self.apply_im(sample[0]['im'], sample[1]['im'], factor)
        result = copy.deepcopy(sample[0])
        result['im'] = im
        # apply bbox and score
        if 'gt_bbox' in sample[0]:
            gt_bbox1 = sample[0]['gt_bbox']
            gt_bbox2 = sample[1]['gt_bbox']
            gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
            result['gt_bbox'] = gt_bbox
        if 'gt_class' in sample[0]:
            gt_class1 = sample[0]['gt_class']
            gt_class2 = sample[1]['gt_class']
            gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
            result['gt_class'] = gt_class

            gt_score1 = np.ones_like(sample[0]['gt_class'])
            gt_score2 = np.ones_like(sample[1]['gt_class'])
            gt_score = np.concatenate(
                (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
            result['gt_score'] = gt_score
        if 'is_crowd' in sample[0]:
            is_crowd1 = sample[0]['is_crowd']
            is_crowd2 = sample[1]['is_crowd']
            is_crowd = np.concatenate((is_crowd1, is_crowd2), axis=0)
            result['is_crowd'] = is_crowd
        if 'difficult' in sample[0]:
            is_difficult1 = sample[0]['difficult']
            is_difficult2 = sample[1]['difficult']
            is_difficult = np.concatenate(
                (is_difficult1, is_difficult2), axis=0)
            result['difficult'] = is_difficult

        return result


class ArrangeSegmenter(Transform):
    def __init__(self, mode):
        super(ArrangeSegmenter, self).__init__()
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, sample):
        im = sample['im']
        if 'mask' in sample:
            mask = sample['mask']

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
        super(ArrangeClassifier, self).__init__()
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def __call__(self, sample):
        im = sample['im']
        im = permute(im, False)
        return im, sample['label']
