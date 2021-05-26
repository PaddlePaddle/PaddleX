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
from .operators import Transform, Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, \
    ResizeByShort, Normalize, RandomDistort, ArrangeSegmenter
from .operators import Padding as dy_Padding


class ResizeByLong(Transform):
    """对图像长边resize到固定值，短边按比例进行缩放。当存在标注图像时，则同步进行处理。
    Args:
        long_size (int): resize后图像的长边大小。
    """

    def __init__(self, long_size=256):
        super(ResizeByLong, self).__init__()
        self.long_size = long_size

    def apply_im(self, image):
        image = _resize_long(image, long_size=self.long_size)
        return image

    def apply_mask(self, mask):
        mask = _resize_long(
            mask, long_size=self.long_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'])

        return sample


class ResizeRangeScaling(Transform):
    """对图像长边随机resize到指定范围内，短边按比例进行缩放。当存在标注图像时，则同步进行处理。
    Args:
        min_value (int): 图像长边resize后的最小值。默认值400。
        max_value (int): 图像长边resize后的最大值。默认值600。
    Raises:
        ValueError: min_value大于max_value
    """

    def __init__(self, min_value=400, max_value=600):
        super(ResizeRangeScaling, self).__init__()
        if min_value > max_value:
            raise ValueError('min_value must be less than max_value, '
                             'but they are {} and {}.'.format(min_value,
                                                              max_value))
        self.min_value = min_value
        self.max_value = max_value

    def apply_im(self, image, random_size):
        image = _resize_long(image, long_size=random_size)
        return image

    def apply_mask(self, mask, random_size):
        mask = _resize_long(
            mask, long_size=random_size, interpolation=cv2.INTER_NEAREST)
        return mask

    def apply(self, sample):
        if self.min_value == self.max_value:
            random_size = self.max_value
        else:
            random_size = int(
                np.random.uniform(self.min_value, self.max_value) + 0.5)
        sample['image'] = self.apply_im(sample['image'], random_size)
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], random_size)

        return sample


class ResizeStepScaling(Transform):
    """对图像按照某一个比例resize，这个比例以scale_step_size为步长
    在[min_scale_factor, max_scale_factor]随机变动。当存在标注图像时，则同步进行处理。
    Args:
        min_scale_factor（float), resize最小尺度。默认值0.75。
        max_scale_factor (float), resize最大尺度。默认值1.25。
        scale_step_size (float), resize尺度范围间隔。默认值0.25。
    Raises:
        ValueError: min_scale_factor大于max_scale_factor
    """

    def __init__(self,
                 min_scale_factor=0.75,
                 max_scale_factor=1.25,
                 scale_step_size=0.25):
        if min_scale_factor > max_scale_factor:
            raise ValueError(
                'min_scale_factor must be less than max_scale_factor, '
                'but they are {} and {}.'.format(min_scale_factor,
                                                 max_scale_factor))
        super(ResizeStepScaling, self).__init__()
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_step_size = scale_step_size

    def apply_im(self, image, scale_factor):
        image = cv2.resize(
            image, (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_LINEAR)
        if image.ndim < 3:
            image = np.expand_dims(image, axis=-1)
        return image

    def apply_mask(self, mask, scale_factor):
        mask = cv2.resize(
            mask, (0, 0),
            fx=scale_factor,
            fy=scale_factor,
            interpolation=cv2.INTER_NEAREST)
        return mask

    def apply(self, sample):
        if self.min_scale_factor == self.max_scale_factor:
            scale_factor = self.min_scale_factor

        elif self.scale_step_size == 0:
            scale_factor = np.random.uniform(self.min_scale_factor,
                                             self.max_scale_factor)

        else:
            num_steps = int((self.max_scale_factor - self.min_scale_factor) /
                            self.scale_step_size + 1)
            scale_factors = np.linspace(self.min_scale_factor,
                                        self.max_scale_factor,
                                        num_steps).tolist()
            np.random.shuffle(scale_factors)
            scale_factor = scale_factors[0]

        sample['image'] = self.apply_im(sample['image'], scale_factor)
        if 'mask' in sample:
            sample['mask'] = self.apply_mask(sample['mask'], scale_factor)

        return sample


class Padding(dy_Padding):
    """对图像或标注图像进行padding，padding方向为右和下。
    根据提供的值对图像或标注图像进行padding操作。
    Args:
        target_size (int|list|tuple): padding后图像的大小。
        im_padding_value (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
        label_padding_value (int): 标注图像padding的值。默认值为255。
    Raises:
        TypeError: target_size不是int|list|tuple。
        ValueError:  target_size为list|tuple时元素个数不等于2。
    """

    def __init__(self,
                 target_size,
                 im_padding_value=[127.5, 127.5, 127.5],
                 label_padding_value=255):
        super(Padding, self).__init__(
            target_size=target_size,
            pad_mode=0,
            offsets=None,
            im_padding_value=im_padding_value,
            label_padding_value=label_padding_value)


class RandomPaddingCrop(Transform):
    """对图像和标注图进行随机裁剪，当所需要的裁剪尺寸大于原图时，则进行padding操作。
    Args:
        crop_size (int|list|tuple): 裁剪图像大小。默认为512。
        im_padding_value (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
        label_padding_value (int): 标注图像padding的值。默认值为255。
    Raises:
        TypeError: crop_size不是int/list/tuple。
        ValueError:  target_size为list/tuple时元素个数不等于2。
    """

    def __init__(self,
                 crop_size=512,
                 im_padding_value=[127.5, 127.5, 127.5],
                 label_padding_value=255):
        if isinstance(crop_size, list) or isinstance(crop_size, tuple):
            if len(crop_size) != 2:
                raise ValueError(
                    'when crop_size is list or tuple, it should include 2 elements, but it is {}'
                    .format(crop_size))
        elif not isinstance(crop_size, int):
            raise TypeError(
                "Type of crop_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(crop_size)))
        super(RandomPaddingCrop, self).__init__()
        self.crop_size = crop_size
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply_im(self, image, pad_h, pad_w):
        im_h, im_w, im_c = image.shape
        orig_im = copy.deepcopy(image)
        image = np.zeros(
            (im_h + pad_h, im_w + pad_w, im_c)).astype(orig_im.dtype)
        for i in range(im_c):
            image[:, :, i] = np.pad(orig_im[:, :, i],
                                    pad_width=((0, pad_h), (0, pad_w)),
                                    mode='constant',
                                    constant_values=(self.im_padding_value[i],
                                                     self.im_padding_value[i]))
        return image

    def apply_mask(self, mask, pad_h, pad_w):
        mask = np.pad(mask,
                      pad_width=((0, pad_h), (0, pad_w)),
                      mode='constant',
                      constant_values=(self.label_padding_value,
                                       self.label_padding_value))
        return mask

    def apply(self, sample):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (list): 存储图像reisze或padding前的shape信息，如
                [('resize', [200, 300]), ('padding', [400, 600])]表示
                图像在过resize前shape为(200, 300)， 过padding前shape为
                (400, 600)
            label (np.ndarray): 标注图像np.ndarray数据。
         Returns:
            tuple: 当label为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                当label不为空时，返回的tuple为(im, im_info, label)，分别对应图像np.ndarray数据、
                存储与图像相关信息的字典和标注图像np.ndarray数据。
        """
        if isinstance(self.crop_size, int):
            crop_width = self.crop_size
            crop_height = self.crop_size
        else:
            crop_width = self.crop_size[0]
            crop_height = self.crop_size[1]

        im_h, im_w, im_c = sample['image'].shape

        if im_h == crop_height and im_w == crop_width:
            return sample
        else:
            pad_height = max(crop_height - im_h, 0)
            pad_width = max(crop_width - im_w, 0)
            if pad_height > 0 or pad_width > 0:
                sample['image'] = self.apply_im(sample['image'], pad_height,
                                                pad_width)

                if 'mask' in sample:
                    sample['mask'] = self.apply_mask(sample['mask'],
                                                     pad_height, pad_width)

                im_h = sample['image'].shape[0]
                im_w = sample['image'].shape[1]

            if crop_height > 0 and crop_width > 0:
                h_off = np.random.randint(im_h - crop_height + 1)
                w_off = np.random.randint(im_w - crop_width + 1)

                sample['image'] = sample['image'][h_off:(
                    crop_height + h_off), w_off:(w_off + crop_width), :]
                if 'mask' in sample:
                    sample['mask'] = sample['mask'][h_off:(
                        crop_height + h_off), w_off:(w_off + crop_width)]
        return sample


class RandomBlur(Transform):
    """以一定的概率对图像进行高斯模糊。
    Args：
        prob (float): 图像模糊概率。默认为0.1。
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


class RandomRotate(Transform):
    """对图像进行随机旋转, 模型训练时的数据增强操作。
    在旋转区间[-rotate_range, rotate_range]内，对图像进行随机旋转，当存在标注图像时，同步进行，
    并对旋转后的图像和标注图像进行相应的padding。
    Args:
        rotate_range (float): 最大旋转角度。默认为15度。
        im_padding_value (list): 图像padding的值。默认为[127.5, 127.5, 127.5]。
        label_padding_value (int): 标注图像padding的值。默认为255。
    """

    def __init__(self,
                 rotate_range=15,
                 im_padding_value=[127.5, 127.5, 127.5],
                 label_padding_value=255):
        super(RandomRotate, self).__init__()
        self.rotate_range = rotate_range
        self.im_padding_value = im_padding_value
        self.label_padding_value = label_padding_value

    def apply(self, sample):
        if self.rotate_range > 0:
            h, w, c = sample['image'].shape
            do_rotation = np.random.uniform(-self.rotate_range,
                                            self.rotate_range)
            pc = (w // 2, h // 2)
            r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
            cos = np.abs(r[0, 0])
            sin = np.abs(r[0, 1])

            nw = int((h * sin) + (w * cos))
            nh = int((h * cos) + (w * sin))

            (cx, cy) = pc
            r[0, 2] += (nw / 2) - cx
            r[1, 2] += (nh / 2) - cy
            dsize = (nw, nh)
            rot_ims = list()
            for i in range(0, c, 3):
                ori_im = sample['image'][:, :, i:i + 3]
                rot_im = cv2.warpAffine(
                    ori_im,
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.im_padding_value[i:i + 3])
                rot_ims.append(rot_im)
            sample['image'] = np.concatenate(rot_ims, axis=-1)
            if 'mask' in sample:
                sample['mask'] = cv2.warpAffine(
                    sample['mask'],
                    r,
                    dsize=dsize,
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=self.label_padding_value)

        return sample


class RandomScaleAspect(Transform):
    """裁剪并resize回原始尺寸的图像和标注图像。
    按照一定的面积比和宽高比对图像进行裁剪，并reszie回原始图像的图像，当存在标注图时，同步进行。
    Args：
        min_scale (float)：裁取图像占原始图像的面积比，取值[0，1]，为0时则返回原图。默认为0.5。
        aspect_ratio (float): 裁取图像的宽高比范围，非负值，为0时返回原图。默认为0.33。
    """

    def __init__(self, min_scale=0.5, aspect_ratio=0.33):
        super(RandomScaleAspect, self).__init__()
        self.min_scale = min_scale
        self.aspect_ratio = aspect_ratio

    def apply(self, sample):
        if self.min_scale != 0 and self.aspect_ratio != 0:
            img_height = sample['image'].shape[0]
            img_width = sample['image'].shape[1]
            for i in range(0, 10):
                area = img_height * img_width
                target_area = area * np.random.uniform(self.min_scale, 1.0)
                aspectRatio = np.random.uniform(self.aspect_ratio,
                                                1.0 / self.aspect_ratio)

                dw = int(np.sqrt(target_area * 1.0 * aspectRatio))
                dh = int(np.sqrt(target_area * 1.0 / aspectRatio))
                if (np.random.randint(10) < 5):
                    tmp = dw
                    dw = dh
                    dh = tmp

                if (dh < img_height and dw < img_width):
                    h1 = np.random.randint(0, img_height - dh)
                    w1 = np.random.randint(0, img_width - dw)

                    sample['image'] = sample['image'][h1:(h1 + dh), w1:(w1 + dw
                                                                        ), :]
                    sample['image'] = cv2.resize(
                        sample['image'], (img_width, img_height),
                        interpolation=cv2.INTER_LINEAR)
                    if sample['image'].ndim < 3:
                        sample['image'] = np.expand_dims(
                            sample['image'], axis=-1)

                    if 'mask' in sample:
                        sample['mask'] = sample['mask'][h1:(h1 + dh), w1:(w1 +
                                                                          dw)]
                        sample['mask'] = cv2.resize(
                            sample['mask'], (img_width, img_height),
                            interpolation=cv2.INTER_NEAREST)
                    break
        return sample


class Clip(Transform):
    """
    对图像上超出一定范围的数据进行截断。
    Args:
        min_val (list): 裁剪的下限，小于min_val的数值均设为min_val. 默认值0.
        max_val (list): 裁剪的上限，大于max_val的数值均设为max_val. 默认值255.0.
    """

    def __init__(self, min_val=[0, 0, 0], max_val=[255.0, 255.0, 255.0]):
        if not (isinstance(min_val, list) and isinstance(max_val, list)):
            raise ValueError("{}: input type is invalid.".format(self))
        super(Clip, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def apply_im(self, image):
        for k in range(image.shape[2]):
            np.clip(
                image[:, :, k],
                self.min_val[k],
                self.max_val[k],
                out=image[:, :, k])
        return image

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        return sample


class ComposedSegTransforms(Compose):
    """ 语义分割模型(UNet/DeepLabv3p)的图像处理流程，具体如下
        训练阶段：
        1. 随机对图像以0.5的概率水平翻转，若random_horizontal_flip为False，则跳过此步骤
        2. 按不同的比例随机Resize原图, 处理方式参考[paddlex.seg.transforms.ResizeRangeScaling](#resizerangescaling)。若min_max_size为None，则跳过此步骤
        3. 从原图中随机crop出大小为train_crop_size大小的子图，如若crop出来的图小于train_crop_size，则会将图padding到对应大小
        4. 图像归一化
       预测阶段：
        1. 将图像的最长边resize至(min_max_size[0] + min_max_size[1])//2, 短边按比例resize。若min_max_size为None，则跳过此步骤
        2. 图像归一化
        Args:
            mode(str): Transforms所处的阶段，包括`train', 'eval'或'test'
            min_max_size(list): 用于对图像进行resize，具体作用参见上述步骤。
            train_crop_size(list): 训练过程中随机裁剪原图用于训练，具体作用参见上述步骤。此参数仅在mode为`train`时生效。
            mean(list): 图像均值, 默认为[0.485, 0.456, 0.406]。
            std(list): 图像方差，默认为[0.229, 0.224, 0.225]。
            random_horizontal_flip(bool): 数据增强，是否随机水平翻转图像，此参数仅在mode为`train`时生效。
    """

    def __init__(self,
                 mode,
                 min_max_size=[400, 600],
                 train_crop_size=[512, 512],
                 mean=[0.5, 0.5, 0.5],
                 std=[0.5, 0.5, 0.5],
                 random_horizontal_flip=True):
        if mode == 'train':
            # 训练时的transforms，包含数据增强
            if min_max_size is None:
                transforms = [
                    RandomPaddingCrop(crop_size=train_crop_size), Normalize(
                        mean=mean, std=std)
                ]
            else:
                transforms = [
                    ResizeRangeScaling(
                        min_value=min(min_max_size),
                        max_value=max(min_max_size)),
                    RandomPaddingCrop(crop_size=train_crop_size), Normalize(
                        mean=mean, std=std)
                ]
            if random_horizontal_flip:
                transforms.insert(0, RandomHorizontalFlip())
        else:
            # 验证/预测时的transforms
            if min_max_size is None:
                transforms = [Normalize(mean=mean, std=std)]
            else:
                long_size = (min(min_max_size) + max(min_max_size)) // 2
                transforms = [
                    ResizeByLong(long_size=long_size), Normalize(
                        mean=mean, std=std)
                ]
        super(ComposedSegTransforms, self).__init__(transforms)


def _resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im_dims = im.ndim
    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    if im_dims >= 3 and im.ndim < 3:
        im = np.expand_dims(im, axis=-1)
    return im
