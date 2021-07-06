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
import paddlex

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
from numbers import Number
from .functions import normalize, horizontal_flip, permute, vertical_flip, center_crop, is_poly, \
    horizontal_flip_poly, horizontal_flip_rle, vertical_flip_poly, vertical_flip_rle, crop_poly, \
    crop_rle, expand_poly, expand_rle, resize_poly, resize_rle

__all__ = [
    "Compose", "Decode", "Resize", "RandomResize", "ResizeByShort",
    "RandomResizeByShort", "ResizeByLong", "RandomHorizontalFlip",
    "RandomVerticalFlip", "Normalize", "CenterCrop", "RandomCrop",
    "RandomScaleAspect", "RandomExpand", "Padding", "MixupImage",
    "RandomDistort", "RandomBlur", "ArrangeSegmenter", "ArrangeClassifier",
    "ArrangeDetector"
]

interp_dict = {
    'NEAREST': cv2.INTER_NEAREST,
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'AREA': cv2.INTER_AREA,
    'LANCZOS4': cv2.INTER_LANCZOS4
}


class Transform(object):
    """
    Parent class of all data augmentation operations
    """

    def __init__(self):
        pass

    def apply_im(self, image):
        pass

    def apply_mask(self, mask):
        pass

    def apply_bbox(self, bbox):
        pass

    def apply_segm(self, segms):
        pass

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        if 'gt_bbox' in sample:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'])

        return sample

    def __call__(self, sample):
        if isinstance(sample, Sequence):
            sample = [self.apply(s) for s in sample]
        else:
            sample = self.apply(sample)

        return sample


class Compose(Transform):
    """
    Apply a series of data augmentation to the input.
    All input images are in Height-Width-Channel ([H, W, C]) format.

    Args:
        transforms (List[paddlex.transforms.Transform]): List of data preprocess or augmentations.
    Raises:
        TypeError: Invalid type of transforms.
        ValueError: Invalid length of transforms.
    """

    def __init__(self, transforms):
        super(Compose, self).__init__()
        if not isinstance(transforms, list):
            raise TypeError(
                'Type of transforms is invalid. Must be List, but received is {}'
                .format(type(transforms)))
        if len(transforms) < 1:
            raise ValueError(
                'Length of transforms must not be less than 1, but received is {}'
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
            # skip batch transforms amd mixup
            if isinstance(op, (paddlex.transforms.BatchRandomResize,
                               paddlex.transforms.BatchRandomResizeByShort,
                               MixupImage)):
                continue
            sample = op(sample)

        if self.arrange_outputs is not None:
            if self.apply_im_only:
                sample['mask'] = mask_backup
            sample = self.arrange_outputs(sample)

        return sample


class Decode(Transform):
    """
    Decode image(s) in input.

    Args:
        to_rgb (bool, optional): If True, convert input images from BGR format to RGB format. Defaults to True.
    """

    def __init__(self, to_rgb=True):
        super(Decode, self).__init__()
        self.to_rgb = to_rgb

    def read_img(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR |
                          cv2.IMREAD_COLOR)

    def apply_im(self, im_path):
        if isinstance(im_path, str):
            try:
                image = self.read_img(im_path)
            except:
                raise ValueError('Cannot read the image file {}!'.format(
                    im_path))
        else:
            image = im_path

        if self.to_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

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

    def apply(self, sample):
        """

        Args:
            sample (dict): Input sample, containing 'image' at least.

        Returns:
            dict: Decoded sample.

        """
        sample['image'] = self.apply_im(sample['image'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
            im_height, im_width, _ = sample['image'].shape
            se_height, se_width = sample['mask'].shape
            if im_height != se_height or im_width != se_width:
                raise Exception(
                    "The height or width of the im is not same as the mask")
        sample['im_shape'] = np.array(
            sample['image'].shape[:2], dtype=np.float32)
        sample['scale_factor'] = np.array([1., 1.], dtype=np.float32)
        return sample


class Resize(Transform):
    """
    Resize input.

    - If target_size is an int，resize the image(s) to (target_size, target_size).
    - If target_size is a list or tuple, resize the image(s) to target_size.
    Attention：If interp is 'RANDOM', the interpolation method will be chose randomly.

    Args:
        target_size (int, List[int] or Tuple[int]): Target size. If int, the height and width share the same target_size.
            Otherwise, target_size represents [target height, target width].
        interp ({'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}, optional):
            Interpolation method of resize. Defaults to 'LINEAR'.
        keep_ratio (bool): the resize scale of width/height is same and width/height after resized is not greater
            than target width/height. Defaults to False.

    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.
    """

    def __init__(self, target_size, interp='LINEAR', keep_ratio=False):
        super(Resize, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("interp should be one of {}".format(
                interp_dict.keys()))
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        else:
            if not (isinstance(target_size,
                               (list, tuple)) and len(target_size) == 2):
                raise TypeError(
                    "target_size should be an int or a list of length 2, but received {}".
                    format(target_size))
        # (height, width)
        self.target_size = target_size
        self.interp = interp
        self.keep_ratio = keep_ratio

    def apply_im(self, image, interp, target_size):
        image = cv2.resize(image, target_size, interpolation=interp)
        return image

    def apply_mask(self, mask, target_size):
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def apply_bbox(self, bbox, scale, target_size):
        im_scale_x, im_scale_y = scale
        bbox[:, 0::2] *= im_scale_x
        bbox[:, 1::2] *= im_scale_y
        bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, target_size[0])
        bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, target_size[1])
        return bbox

    def apply_segm(self, segms, im_size, scale):
        im_h, im_w = im_size
        im_scale_x, im_scale_y = scale
        resized_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                resized_segms.append([
                    resize_poly(poly, im_scale_x, im_scale_y) for poly in segm
                ])
            else:
                # RLE format
                resized_segms.append(
                    resize_rle(segm, im_h, im_w, im_scale_x, im_scale_y))

        return resized_segms

    def apply(self, sample):
        if self.interp == "RANDOM":
            interp = random.choice(list(interp_dict.values()))
        else:
            interp = interp_dict[self.interp]
        im_h, im_w = sample['image'].shape[:2]

        im_scale_y = self.target_size[0] / im_h
        im_scale_x = self.target_size[1] / im_w
        target_size = (self.target_size[1], self.target_size[0])
        if self.keep_ratio:
            scale = min(im_scale_y, im_scale_x)
            target_w = int(round(im_w * scale))
            target_h = int(round(im_h * scale))
            target_size = (target_w, target_h)
            im_scale_y = target_h / im_h
            im_scale_x = target_w / im_w

        sample['image'] = self.apply_im(sample['image'], interp, target_size)

        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], target_size)
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(
                sample['gt_bbox'], [im_scale_x, im_scale_y], target_size)
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(
                sample['gt_poly'], [im_h, im_w], [im_scale_x, im_scale_y])
        sample['im_shape'] = np.asarray(
            sample['image'].shape[:2], dtype=np.float32)
        if 'scale_factor' in sample:
            scale_factor = sample['scale_factor']
            sample['scale_factor'] = np.asarray(
                [scale_factor[0] * im_scale_y, scale_factor[1] * im_scale_x],
                dtype=np.float32)

        return sample


class RandomResize(Transform):
    """
    Resize input to random sizes.

    Attention：If interp is 'RANDOM', the interpolation method will be chose randomly.

    Args:
        target_sizes (List[int], List[list or tuple] or Tuple[list or tuple]):
            Multiple target sizes, each target size is an int or list/tuple.
        interp ({'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}, optional):
            Interpolation method of resize. Defaults to 'LINEAR'.

    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.

    See Also:
        Resize input to a specific size.
    """

    def __init__(self, target_sizes, interp='LINEAR'):
        super(RandomResize, self).__init__()
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

    def apply(self, sample):
        height, width = random.choice(self.target_size)
        resizer = Resize((height, width), interp=self.interp)
        sample = resizer(sample)

        return sample


class ResizeByShort(Transform):
    """
    Resize input with keeping the aspect ratio.

    Attention：If interp is 'RANDOM', the interpolation method will be chose randomly.

    Args:
        short_size (int): Target size of the shorter side of the image(s).
        max_size (int, optional): The upper bound of longer side of the image(s). If max_size is -1, no upper bound is applied. Defaults to -1.
        interp ({'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}, optional): Interpolation method of resize. Defaults to 'LINEAR'.

    Raises:
        ValueError: Invalid interpolation method.
    """

    def __init__(self, short_size=256, max_size=-1, interp='LINEAR'):
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("interp should be one of {}".format(
                interp_dict.keys()))
        super(ResizeByShort, self).__init__()
        self.short_size = short_size
        self.max_size = max_size
        self.interp = interp

    def apply(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        im_short_size = min(im_h, im_w)
        im_long_size = max(im_h, im_w)
        scale = float(self.short_size) / float(im_short_size)
        if 0 < self.max_size < np.round(scale * im_long_size):
            scale = float(self.max_size) / float(im_long_size)
        target_w = int(round(im_w * scale))
        target_h = int(round(im_h * scale))
        target_size = (target_w, target_h)
        sample = Resize(target_size=target_size, interp=self.interp)(sample)

        return sample


class RandomResizeByShort(Transform):
    """
    Resize input to random sizes with keeping the aspect ratio.

    Attention：If interp is 'RANDOM', the interpolation method will be chose randomly.

    Args:
        short_sizes (List[int]): Target size of the shorter side of the image(s).
        max_size (int, optional): The upper bound of longer side of the image(s). If max_size is -1, no upper bound is applied. Defaults to -1.
        interp ({'NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM'}, optional): Interpolation method of resize. Defaults to 'LINEAR'.

    Raises:
        TypeError: Invalid type of target_size.
        ValueError: Invalid interpolation method.

    See Also:
        ResizeByShort: Resize image(s) in input with keeping the aspect ratio.
    """

    def __init__(self, short_sizes, max_size=-1, interp='LINEAR'):
        super(RandomResizeByShort, self).__init__()
        if not (interp == "RANDOM" or interp in interp_dict):
            raise ValueError("interp should be one of {}".format(
                interp_dict.keys()))
        self.interp = interp
        assert isinstance(short_sizes, list), \
            "short_sizes must be List"

        self.short_sizes = short_sizes
        self.max_size = max_size

    def apply(self, sample):
        short_size = random.choice(self.short_sizes)
        resizer = ResizeByShort(
            short_size=short_size, max_size=self.max_size, interp=self.interp)
        sample = resizer(sample)
        return sample


class ResizeByLong(Transform):
    def __init__(self, long_size=256, interp='LINEAR'):
        super(ResizeByLong, self).__init__()
        self.long_size = long_size
        self.interp = interp

    def apply(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        im_long_size = max(im_h, im_w)
        scale = float(self.long_size) / float(im_long_size)
        target_h = int(round(im_h * scale))
        target_w = int(round(im_w * scale))
        target_size = (target_w, target_h)
        sample = Resize(target_size=target_size, interp=self.interp)(sample)

        return sample


class RandomHorizontalFlip(Transform):
    """
    Randomly flip the input horizontally.

    Args:
        prob(float, optional): Probability of flipping the input. Defaults to .5.
    """

    def __init__(self, prob=0.5):
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        image = horizontal_flip(image)
        return image

    def apply_mask(self, mask):
        mask = horizontal_flip(mask)
        return mask

    def apply_bbox(self, bbox, width):
        oldx1 = bbox[:, 0].copy()
        oldx2 = bbox[:, 2].copy()
        bbox[:, 0] = width - oldx2
        bbox[:, 2] = width - oldx1
        return bbox

    def apply_segm(self, segms, height, width):
        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append(
                    [horizontal_flip_poly(poly, width) for poly in segm])
            else:
                # RLE format
                flipped_segms.append(horizontal_flip_rle(segm, height, width))
        return flipped_segms

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_w)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h,
                                                    im_w)
        return sample


class RandomVerticalFlip(Transform):
    """
    Randomly flip the input vertically.

    Args:
        prob(float, optional): Probability of flipping the input. Defaults to .5.
    """

    def __init__(self, prob=0.5):
        super(RandomVerticalFlip, self).__init__()
        self.prob = prob

    def apply_im(self, image):
        image = vertical_flip(image)
        return image

    def apply_mask(self, mask):
        mask = vertical_flip(mask)
        return mask

    def apply_bbox(self, bbox, height):
        oldy1 = bbox[:, 1].copy()
        oldy2 = bbox[:, 3].copy()
        bbox[:, 0] = height - oldy2
        bbox[:, 2] = height - oldy1
        return bbox

    def apply_segm(self, segms, height, width):
        flipped_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                flipped_segms.append(
                    [vertical_flip_poly(poly, height) for poly in segm])
            else:
                # RLE format
                flipped_segms.append(vertical_flip_rle(segm, height, width))
        return flipped_segms

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'])
            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'])
            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], im_h)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                sample['gt_poly'] = self.apply_segm(sample['gt_poly'], im_h,
                                                    im_w)
        return sample


class Normalize(Transform):
    """
    Apply min-max normalization to the image(s) in input.
    1. im = (im - min_value) * 1 / (max_value - min_value)
    2. im = im - mean
    3. im = im / std

    Args:
        mean(List[float] or Tuple[float], optional): Mean of input image(s). Defaults to [0.485, 0.456, 0.406].
        std(List[float] or Tuple[float], optional): Standard deviation of input image(s). Defaults to [0.229, 0.224, 0.225].
        min_val(List[float] or Tuple[float], optional): Minimum value of input image(s). Defaults to [0, 0, 0, ].
        max_val(List[float] or Tuple[float], optional): Max value of input image(s). Defaults to [255., 255., 255.].
        is_scale(bool, optional): If True, the image pixel values will be divided by 255.
    """

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

    def apply_im(self, image):
        image = image.astype(np.float32)
        mean = np.asarray(
            self.mean, dtype=np.float32)[np.newaxis, np.newaxis, :]
        std = np.asarray(self.std, dtype=np.float32)[np.newaxis, np.newaxis, :]
        image = normalize(image, mean, std, self.min_val, self.max_val)
        return image

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])

        return sample


class CenterCrop(Transform):
    """
    Crop the input at the center.
    1. Locate the center of the image.
    2. Crop the sample.

    Args:
        crop_size(int, optional): target size of the cropped image(s). Defaults to 224.
    """

    def __init__(self, crop_size=224):
        super(CenterCrop, self).__init__()
        self.crop_size = crop_size

    def apply_im(self, image):
        image = center_crop(image, self.crop_size)

        return image

    def apply_mask(self, mask):
        mask = center_crop(mask)
        return mask

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])
        return sample


class RandomCrop(Transform):
    """
    Randomly crop the input.
    1. Compute the height and width of cropped area according to aspect_ratio and scaling.
    2. Locate the upper left corner of cropped area randomly.
    3. Crop the image(s).
    4. Resize the cropped area to crop_size by crop_size.

    Args:
        crop_size(int, List[int] or Tuple[int]): Target size of the cropped area. If None, the cropped area will not be
            resized. Defaults to None.
        aspect_ratio (List[float], optional): Aspect ratio of cropped region in [min, max] format. Defaults to [.5, 2.].
        thresholds (List[float], optional): Iou thresholds to decide a valid bbox crop.
            Defaults to [.0, .1, .3, .5, .7, .9].
        scaling (List[float], optional): Ratio between the cropped region and the original image in [min, max] format.
            Defaults to [.3, 1.].
        num_attempts (int, optional): The number of tries before giving up. Defaults to 50.
        allow_no_crop (bool, optional): Whether returning without doing crop is allowed. Defaults to True.
        cover_all_box (bool, optional): Whether to ensure all bboxes are covered in the final crop. Defaults to False.
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

    def _generate_crop_info(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            thresholds = self.thresholds
            if self.allow_no_crop:
                thresholds.append('no_crop')
            np.random.shuffle(thresholds)
            for thresh in thresholds:
                if thresh == 'no_crop':
                    return None
                for i in range(self.num_attempts):
                    crop_box = self._get_crop_box(im_h, im_w)
                    if crop_box is None:
                        continue
                    iou = self._iou_matrix(
                        sample['gt_bbox'],
                        np.array(
                            [crop_box], dtype=np.float32))
                    if iou.max() < thresh:
                        continue
                    if self.cover_all_box and iou.min() < thresh:
                        continue
                    cropped_box, valid_ids = self._crop_box_with_center_constraint(
                        sample['gt_bbox'],
                        np.array(
                            crop_box, dtype=np.float32))
                    if valid_ids.size > 0:
                        return crop_box, cropped_box, valid_ids
        else:
            for i in range(self.num_attempts):
                crop_box = self._get_crop_box(im_h, im_w)
                if crop_box is None:
                    continue
                return crop_box, None, None
        return None

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

    def _crop_segm(self, segms, valid_ids, crop, height, width):
        crop_segms = []
        for id in valid_ids:
            segm = segms[id]
            if is_poly(segm):
                # Polygon format
                crop_segms.append(crop_poly(segm, crop))
            else:
                # RLE format
                crop_segms.append(crop_rle(segm, crop, height, width))

        return crop_segms

    def apply_im(self, image, crop):
        x1, y1, x2, y2 = crop
        return image[y1:y2, x1:x2, :]

    def apply_mask(self, mask, crop):
        x1, y1, x2, y2 = crop
        return mask[y1:y2, x1:x2, ...]

    def apply(self, sample):
        crop_info = self._generate_crop_info(sample)
        if crop_info is not None:
            crop_box, cropped_box, valid_ids = crop_info
            im_h, im_w = sample['image'].shape[:2]
            sample['image'] = self.apply_im(sample['image'], crop_box)
            if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
                crop_polys = self._crop_segm(
                    sample['gt_poly'],
                    valid_ids,
                    np.array(
                        crop_box, dtype=np.int64),
                    im_h,
                    im_w)
                if [] in crop_polys:
                    delete_id = list()
                    valid_polys = list()
                    for idx, poly in enumerate(crop_polys):
                        if not crop_poly:
                            delete_id.append(idx)
                        else:
                            valid_polys.append(poly)
                    valid_ids = np.delete(valid_ids, delete_id)
                    if not valid_polys:
                        return sample
                    sample['gt_poly'] = valid_polys
                else:
                    sample['gt_poly'] = crop_polys

            if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
                sample['gt_bbox'] = np.take(cropped_box, valid_ids, axis=0)
                sample['gt_class'] = np.take(
                    sample['gt_class'], valid_ids, axis=0)
                if 'gt_score' in sample:
                    sample['gt_score'] = np.take(
                        sample['gt_score'], valid_ids, axis=0)
                if 'is_crowd' in sample:
                    sample['is_crowd'] = np.take(
                        sample['is_crowd'], valid_ids, axis=0)

            if 'mask' in sample:
                sample['mask'] = self.apply_mask(sample['mask'], crop_box)

        if self.crop_size is not None:
            sample = Resize(self.crop_size)(sample)

        return sample


class RandomScaleAspect(Transform):
    """
    Crop input image(s) and resize back to original sizes.
    Args：
        min_scale (float)：Minimum ratio between the cropped region and the original image.
            If 0, image(s) will not be cropped. Defaults to .5.
        aspect_ratio (float): Aspect ratio of cropped region. Defaults to .33.
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        super(RandomScaleAspect, self).__init__()
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def apply(self, sample):
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height, img_width = sample['image'].shape[:2]
            sample = RandomCrop(
                crop_size=(img_height, img_width),
                aspect_ratio=[self.aspect_ratio, 1. / self.aspect_ratio],
                scaling=[self.min_scale, 1.],
                num_attempts=10,
                allow_no_crop=False)(sample)
        return sample


class RandomExpand(Transform):
    """
    Randomly expand the input by padding according to random offsets.

    Args:
        upper_ratio(float, optional): The maximum ratio to which the original image is expanded. Defaults to 4..
        prob(float, optional): The probability of apply expanding. Defaults to .5.
        im_padding_value(List[float] or Tuple[float], optional): RGB filling value for the image. Defaults to (127.5, 127.5, 127.5).
        label_padding_value(int, optional): Filling value for the mask. Defaults to 255.

    See Also:
        paddlex.transforms.Padding
    """

    def __init__(self,
                 upper_ratio=4.,
                 prob=.5,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255):
        super(RandomExpand, self).__init__()
        assert upper_ratio > 1.01, "expand ratio must be larger than 1.01"
        self.upper_ratio = upper_ratio
        self.prob = prob
        assert isinstance(im_padding_value, (Number, Sequence)), \
            "fill value must be either float or sequence"
        if isinstance(im_padding_value, Number):
            im_padding_value = (im_padding_value, ) * 3
        if not isinstance(im_padding_value, tuple):
            im_padding_value = tuple(im_padding_value)
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply(self, sample):
        if random.random() < self.prob:
            im_h, im_w = sample['image'].shape[:2]
            ratio = np.random.uniform(1., self.upper_ratio)
            h = int(im_h * ratio)
            w = int(im_w * ratio)
            if h > im_h and w > im_w:
                y = np.random.randint(0, h - im_h)
                x = np.random.randint(0, w - im_w)
                target_size = (h, w)
                offsets = (x, y)
                sample = Padding(
                    target_size=target_size,
                    pad_mode=-1,
                    offsets=offsets,
                    im_padding_value=self.im_padding_value,
                    label_padding_value=self.label_padding_value)(sample)
        return sample


class Padding(Transform):
    def __init__(self,
                 target_size=None,
                 pad_mode=0,
                 offsets=None,
                 im_padding_value=(127.5, 127.5, 127.5),
                 label_padding_value=255,
                 size_divisor=32):
        """
        Pad image to a specified size or multiple of size_divisor.

        Args:
            target_size(int, Sequence, optional): Image target size, if None, pad to multiple of size_divisor. Defaults to None.
            pad_mode({-1, 0, 1, 2}, optional): Pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. If 1, pad according to center. If 2, only pad left and top. Defaults to 0.
            im_padding_value(Sequence[float]): RGB value of pad area. Defaults to (127.5, 127.5, 127.5).
            label_padding_value(int, optional): Filling value for the mask. Defaults to 255.
            size_divisor(int): Image width and height after padding is a multiple of coarsest_stride.
        """
        super(Padding, self).__init__()
        if isinstance(target_size, (list, tuple)):
            if len(target_size) != 2:
                raise ValueError(
                    '`target_size` should include 2 elements, but it is {}'.
                    format(target_size))
        if isinstance(target_size, int):
            target_size = [target_size] * 2

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.target_size = target_size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.offsets = offsets
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply_im(self, image, offsets, target_size):
        x, y = offsets
        im_h, im_w = image.shape[:2]
        h, w = target_size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.im_padding_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def apply_mask(self, mask, offsets, target_size):
        x, y = offsets
        im_h, im_w = mask.shape[:2]
        h, w = target_size
        canvas = np.ones((h, w), dtype=np.float32)
        canvas *= np.array(self.label_padding_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w] = mask.astype(np.float32)
        return canvas

    def apply_bbox(self, bbox, offsets):
        return bbox + np.array(offsets * 2, dtype=np.float32)

    def apply_segm(self, segms, offsets, im_size, size):
        x, y = offsets
        height, width = im_size
        h, w = size
        expanded_segms = []
        for segm in segms:
            if is_poly(segm):
                # Polygon format
                expanded_segms.append(
                    [expand_poly(poly, x, y) for poly in segm])
            else:
                # RLE format
                expanded_segms.append(
                    expand_rle(segm, x, y, height, width, h, w))
        return expanded_segms

    def apply(self, sample):
        im_h, im_w = sample['image'].shape[:2]
        if self.target_size:
            h, w = self.target_size
            assert (
                    im_h <= h and im_w <= w
            ), 'target size ({}, {}) cannot be less than image size ({}, {})'\
                .format(h, w, im_h, im_w)
        else:
            h = (np.ceil(im_h / self.size_divisor) *
                 self.size_divisor).astype(int)
            w = (np.ceil(im_w / self.size_divisor) *
                 self.size_divisor).astype(int)

        if h == im_h and w == im_w:
            return sample

        if self.pad_mode == -1:
            offsets = self.offsets
        elif self.pad_mode == 0:
            offsets = [0, 0]
        elif self.pad_mode == 1:
            offsets = [(h - im_h) // 2, (w - im_w) // 2]
        else:
            offsets = [h - im_h, w - im_w]

        sample['image'] = self.apply_im(sample['image'], offsets, (h, w))
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], offsets, (h, w))
        if 'gt_bbox' in sample and len(sample['gt_bbox']) > 0:
            sample['gt_bbox'] = self.apply_bbox(sample['gt_bbox'], offsets)
        if 'gt_poly' in sample and len(sample['gt_poly']) > 0:
            sample['gt_poly'] = self.apply_segm(
                sample['gt_poly'], offsets, im_size=[im_h, im_w], size=[h, w])
        return sample


class MixupImage(Transform):
    def __init__(self, alpha=1.5, beta=1.5, mixup_epoch=-1):
        """
        Mixup two images and their gt_bbbox/gt_score.

        Args:
            alpha (float, optional): Alpha parameter of beta distribution. Defaults to 1.5.
            beta (float, optional): Beta parameter of beta distribution. Defaults to 1.5.
        """
        super(MixupImage, self).__init__()
        if alpha <= 0.0:
            raise ValueError("alpha should be positive in {}".format(self))
        if beta <= 0.0:
            raise ValueError("beta should be positive in {}".format(self))
        self.alpha = alpha
        self.beta = beta
        self.mixup_epoch = mixup_epoch

    def apply_im(self, image1, image2, factor):
        h = max(image1.shape[0], image2.shape[0])
        w = max(image1.shape[1], image2.shape[1])
        img = np.zeros((h, w, image1.shape[2]), 'float32')
        img[:image1.shape[0], :image1.shape[1], :] = \
            image1.astype('float32') * factor
        img[:image2.shape[0], :image2.shape[1], :] += \
            image2.astype('float32') * (1.0 - factor)
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
        image = self.apply_im(sample[0]['image'], sample[1]['image'], factor)
        result = copy.deepcopy(sample[0])
        result['image'] = image
        # apply bbox and score
        if 'gt_bbox' in sample[0]:
            gt_bbox1 = sample[0]['gt_bbox']
            gt_bbox2 = sample[1]['gt_bbox']
            gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
            result['gt_bbox'] = gt_bbox
        if 'gt_poly' in sample[0]:
            gt_poly1 = sample[0]['gt_poly']
            gt_poly2 = sample[1]['gt_poly']
            gt_poly = gt_poly1 + gt_poly2
            result['gt_poly'] = gt_poly
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


class RandomDistort(Transform):
    """
    Random color distortion.

    Args:
        brightness_range(float, optional): Range of brightness distortion. Defaults to .5.
        brightness_prob(float, optional): Probability of brightness distortion. Defaults to .5.
        contrast_range(float, optional): Range of contrast distortion. Defaults to .5.
        contrast_prob(float, optional): Probability of contrast distortion. Defaults to .5.
        saturation_range(float, optional): Range of saturation distortion. Defaults to .5.
        saturation_prob(float, optional): Probability of saturation distortion. Defaults to .5.
        hue_range(float, optional): Range of hue distortion. Defaults to .5.
        hue_prob(float, optional): Probability of hue distortion. Defaults to .5.
        random_apply (bool, optional): whether to apply in random (yolo) or fixed (SSD)
            order. Defaults to True.
        count (int, optional): the number of doing distortion. Defaults to 4.
        shuffle_channel (bool, optional): whether to swap channels randomly. Defaults to False.
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5,
                 random_apply=True,
                 count=4,
                 shuffle_channel=False):
        super(RandomDistort, self).__init__()
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.brightness_prob = brightness_prob
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]
        self.contrast_prob = contrast_prob
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.saturation_prob = saturation_prob
        self.hue_range = [1 - hue_range, 1 + hue_range]
        self.hue_prob = hue_prob
        self.random_apply = random_apply
        self.count = count
        self.shuffle_channel = shuffle_channel

    def apply_hue(self, image):
        low, high = self.hue_range
        if np.random.uniform(0., 1.) < self.hue_prob:
            return image

        image = image.astype(np.float32)
        # it works, but result differ from HSV version
        delta = np.random.uniform(low, high)
        u = np.cos(delta * np.pi)
        w = np.sin(delta * np.pi)
        bt = np.array([[1.0, 0.0, 0.0], [0.0, u, -w], [0.0, w, u]])
        tyiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.274, -0.321],
                         [0.211, -0.523, 0.311]])
        ityiq = np.array([[1.0, 0.956, 0.621], [1.0, -0.272, -0.647],
                          [1.0, -1.107, 1.705]])
        t = np.dot(np.dot(ityiq, bt), tyiq).T
        image = np.dot(image, t)
        return image

    def apply_saturation(self, image):
        low, high = self.saturation_range
        if np.random.uniform(0., 1.) < self.saturation_prob:
            return image
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        # it works, but result differ from HSV version
        gray = image * np.array([[[0.299, 0.587, 0.114]]], dtype=np.float32)
        gray = gray.sum(axis=2, keepdims=True)
        gray *= (1.0 - delta)
        image *= delta
        image += gray
        return image

    def apply_contrast(self, image):
        low, high = self.contrast_range
        if np.random.uniform(0., 1.) < self.contrast_prob:
            return image
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        image *= delta
        return image

    def apply_brightness(self, image):
        low, high = self.brightness_range
        if np.random.uniform(0., 1.) < self.brightness_prob:
            return image
        delta = np.random.uniform(low, high)
        image = image.astype(np.float32)
        image += delta
        return image

    def apply(self, sample):
        if self.random_apply:
            functions = [
                self.apply_brightness, self.apply_contrast,
                self.apply_saturation, self.apply_hue
            ]
            distortions = np.random.permutation(functions)[:self.count]
            for func in distortions:
                sample['image'] = func(sample['image'])
            return sample

        sample['image'] = self.apply_brightness(sample['image'])
        mode = np.random.randint(0, 2)
        if mode:
            sample['image'] = self.apply_contrast(sample['image'])
        sample['image'] = self.apply_saturation(sample['image'])
        sample['image'] = self.apply_hue(sample['image'])
        if not mode:
            sample['image'] = self.apply_contrast(sample['image'])

        if self.shuffle_channel:
            if np.random.randint(0, 2):
                sample['image'] = sample['image'][..., np.random.permutation(
                    3)]

        return sample


class RandomBlur(Transform):
    """
    Randomly blur input image(s).

    Args：
        prob (float): Probability of blurring.
    """

    def __init__(self, prob=0.1):
        super(RandomBlur, self).__init__()
        self.prob = prob

    def apply_im(self, image, radius):
        image = cv2.GaussianBlur(image, (radius, radius), 0, 0)
        return image

    def apply(self, sample):
        if self.prob <= 0:
            n = 0
        elif self.prob >= 1:
            n = 1
        else:
            n = int(1.0 / self.prob)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                sample['image'] = self.apply_im(sample['image'], radius)

        return sample


class _PadBox(Transform):
    def __init__(self, num_max_boxes=50):
        """
        Pad zeros to bboxes if number of bboxes is less than num_max_boxes.

        Args:
            num_max_boxes (int, optional): the max number of bboxes. Defaults to 50.
        """
        self.num_max_boxes = num_max_boxes
        super(_PadBox, self).__init__()

    def apply(self, sample):
        gt_num = min(self.num_max_boxes, len(sample['gt_bbox']))
        num_max = self.num_max_boxes
        pad_bbox = np.zeros((num_max, 4), dtype=np.float32)
        if gt_num > 0:
            pad_bbox[:gt_num, :] = sample['gt_bbox'][:gt_num, :]
        sample['gt_bbox'] = pad_bbox
        if 'gt_class' in sample:
            pad_class = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_class[:gt_num] = sample['gt_class'][:gt_num, 0]
            sample['gt_class'] = pad_class
        if 'gt_score' in sample:
            pad_score = np.zeros((num_max, ), dtype=np.float32)
            if gt_num > 0:
                pad_score[:gt_num] = sample['gt_score'][:gt_num, 0]
            sample['gt_score'] = pad_score
        # in training, for example in op ExpandImage,
        # the bbox and gt_class is expanded, but the difficult is not,
        # so, judging by it's length
        if 'difficult' in sample:
            pad_diff = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_diff[:gt_num] = sample['difficult'][:gt_num, 0]
            sample['difficult'] = pad_diff
        if 'is_crowd' in sample:
            pad_crowd = np.zeros((num_max, ), dtype=np.int32)
            if gt_num > 0:
                pad_crowd[:gt_num] = sample['is_crowd'][:gt_num, 0]
            sample['is_crowd'] = pad_crowd
        return sample


class _NormalizeBox(Transform):
    def __init__(self):
        super(_NormalizeBox, self).__init__()

    def apply(self, sample):
        height, width = sample['image'].shape[:2]
        for i in range(sample['gt_bbox'].shape[0]):
            sample['gt_bbox'][i][0] = sample['gt_bbox'][i][0] / width
            sample['gt_bbox'][i][1] = sample['gt_bbox'][i][1] / height
            sample['gt_bbox'][i][2] = sample['gt_bbox'][i][2] / width
            sample['gt_bbox'][i][3] = sample['gt_bbox'][i][3] / height

        return sample


class _BboxXYXY2XYWH(Transform):
    """
    Convert bbox XYXY format to XYWH format.
    """

    def __init__(self):
        super(_BboxXYXY2XYWH, self).__init__()

    def apply(self, sample):
        bbox = sample['gt_bbox']
        bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, :2]
        bbox[:, :2] = bbox[:, :2] + bbox[:, 2:4] / 2.
        sample['gt_bbox'] = bbox
        return sample


class _Permute(Transform):
    def __init__(self):
        super(_Permute, self).__init__()

    def apply(self, sample):
        sample['image'] = permute(sample['image'], False)
        return sample


class ArrangeSegmenter(Transform):
    def __init__(self, mode):
        super(ArrangeSegmenter, self).__init__()
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def apply(self, sample):
        if 'mask' in sample:
            mask = sample['mask']

        image = permute(sample['image'], False)
        if self.mode == 'train':
            mask = mask.astype('int64')
            return image, mask
        if self.mode == 'eval':
            mask = np.asarray(Image.open(mask))
            mask = mask[np.newaxis, :, :].astype('int64')
            return image, mask
        if self.mode == 'test':
            return image,


class ArrangeClassifier(Transform):
    def __init__(self, mode):
        super(ArrangeClassifier, self).__init__()
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def apply(self, sample):
        image = permute(sample['image'], False)
        if self.mode in ['train', 'eval']:
            return image, sample['label']
        else:
            return image


class ArrangeDetector(Transform):
    def __init__(self, mode):
        super(ArrangeDetector, self).__init__()
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode should be defined as one of ['train', 'eval', 'test', 'quant']!"
            )
        self.mode = mode

    def apply(self, sample):
        if self.mode == 'eval' and 'gt_poly' in sample:
            del sample['gt_poly']
        return sample
