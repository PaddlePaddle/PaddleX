# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .ops import *
from .imgaug_support import execute_imgaug
import random
import os.path as osp
import numpy as np
from PIL import Image, ImageEnhance
import paddlex.utils.logging as logging


class ClsTransform:
    """分类Transform的基类
    """

    def __init__(self):
        pass


class Compose(ClsTransform):
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
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                            'must be equal or larger than 1!')
        self.transforms = transforms
        self.batch_transforms = None
        # 检查transforms里面的操作，目前支持PaddleX定义的或者是imgaug操作
        for op in self.transforms:
            if not isinstance(op, ClsTransform):
                import imgaug.augmenters as iaa
                if not isinstance(op, iaa.Augmenter):
                    raise Exception(
                        "Elements in transforms should be defined in 'paddlex.cls.transforms' or class of imgaug.augmenters.Augmenter, see docs here: https://paddlex.readthedocs.io/zh_CN/latest/apis/transforms/"
                    )

    def __call__(self, im, label=None):
        """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。
        Returns:
            tuple: 根据网络所需字段所组成的tuple；
                字段由transforms中的最后一个数据预处理操作决定。
        """
        if isinstance(im, np.ndarray):
            if len(im.shape) != 3:
                raise Exception(
                    "im should be 3-dimension, but now is {}-dimensions".format(
                        len(im.shape)))
        else:
            try:
                im_path = im
                im = cv2.imread(im).astype('float32')
            except:
                raise TypeError('Can\'t read The image file {}!'.format(
                    im_path))
        im = im.astype('float32')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        for op in self.transforms:
            if isinstance(op, ClsTransform):
                outputs = op(im, label)
                im = outputs[0]
                if len(outputs) == 2:
                    label = outputs[1]
            else:
                import imgaug.augmenters as iaa
                if isinstance(op, iaa.Augmenter):
                    im = execute_imgaug(op, im)
                outputs = (im, )
                if label is not None:
                    outputs = (im, label)
        return outputs

    def add_augmenters(self, augmenters):
        if not isinstance(augmenters, list):
            raise Exception(
                "augmenters should be list type in func add_augmenters()")
        transform_names = [type(x).__name__ for x in self.transforms]
        for aug in augmenters:
            if type(aug).__name__ in transform_names:
                logging.error(
                    "{} is already in ComposedTransforms, need to remove it from add_augmenters().".
                    format(type(aug).__name__))
        self.transforms = augmenters + self.transforms


class RandomCrop(ClsTransform):
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

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        im = random_crop(im, self.crop_size, self.lower_scale, self.lower_ratio,
                         self.upper_ratio)
        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomHorizontalFlip(ClsTransform):
    """以一定的概率对图像进行随机水平翻转，模型训练时的数据增强操作。

    Args:
        prob (float): 随机水平翻转的概率。默认为0.5。
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        if random.random() < self.prob:
            im = horizontal_flip(im)
        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomVerticalFlip(ClsTransform):
    """以一定的概率对图像进行随机垂直翻转，模型训练时的数据增强操作。

    Args:
        prob (float): 随机垂直翻转的概率。默认为0.5。
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        if random.random() < self.prob:
            im = vertical_flip(im)
        if label is None:
            return (im, )
        else:
            return (im, label)


class Normalize(ClsTransform):
    """对图像进行标准化。

    1. 对图像进行归一化到区间[0.0, 1.0]。
    2. 对图像进行减均值除以标准差操作。

    Args:
        mean (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。
        std (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。

    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)
        if label is None:
            return (im, )
        else:
            return (im, label)


class ResizeByShort(ClsTransform):
    """根据图像短边对图像重新调整大小（resize）。

    1. 获取图像的长边和短边长度。
    2. 根据短边与short_size的比例，计算长边的目标长度，
       此时高、宽的resize比例为short_size/原图短边长度。
    3. 如果max_size>0，调整resize比例：
       如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度；
    4. 根据调整大小的比例对图像进行resize。

    Args:
        short_size (int): 调整大小后的图像目标短边长度。默认为256。
        max_size (int): 长边目标长度的最大限制。默认为-1。
    """

    def __init__(self, short_size=256, max_size=-1):
        self.short_size = short_size
        self.max_size = max_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        im_short_size = min(im.shape[0], im.shape[1])
        im_long_size = max(im.shape[0], im.shape[1])
        scale = float(self.short_size) / im_short_size
        if self.max_size > 0 and np.round(scale * im_long_size) > self.max_size:
            scale = float(self.max_size) / float(im_long_size)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im = cv2.resize(
            im, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        if label is None:
            return (im, )
        else:
            return (im, label)


class CenterCrop(ClsTransform):
    """以图像中心点扩散裁剪长宽为`crop_size`的正方形

    1. 计算剪裁的起始点。
    2. 剪裁图像。

    Args:
        crop_size (int): 裁剪的目标边长。默认为224。
    """

    def __init__(self, crop_size=224):
        self.crop_size = crop_size

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        im = center_crop(im, self.crop_size)
        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomRotate(ClsTransform):
    def __init__(self, rotate_range=30, prob=0.5):
        """以一定的概率对图像在[-rotate_range, rotaterange]角度范围内进行旋转，模型训练时的数据增强操作。

        Args:
            rotate_range (int): 旋转度数的范围。默认为30。
            prob (float): 随机旋转的概率。默认为0.5。
        """
        self.rotate_range = rotate_range
        self.prob = prob

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        rotate_lower = -self.rotate_range
        rotate_upper = self.rotate_range
        im = im.astype('uint8')
        im = Image.fromarray(im)
        if np.random.uniform(0, 1) < self.prob:
            im = rotate(im, rotate_lower, rotate_upper)
        im = np.asarray(im).astype('float32')
        if label is None:
            return (im, )
        else:
            return (im, label)


class RandomDistort(ClsTransform):
    """以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作。

    1. 对变换的操作顺序进行随机化操作。
    2. 按照1中的顺序以一定的概率对图像在范围[-range, range]内进行随机像素内容变换。

    Args:
        brightness_range (float): 明亮度因子的范围。默认为0.9。
        brightness_prob (float): 随机调整明亮度的概率。默认为0.5。
        contrast_range (float): 对比度因子的范围。默认为0.9。
        contrast_prob (float): 随机调整对比度的概率。默认为0.5。
        saturation_range (float): 饱和度因子的范围。默认为0.9。
        saturation_prob (float): 随机调整饱和度的概率。默认为0.5。
        hue_range (int): 色调因子的范围。默认为18。
        hue_prob (float): 随机调整色调的概率。默认为0.5。
    """

    def __init__(self,
                 brightness_range=0.9,
                 brightness_prob=0.5,
                 contrast_range=0.9,
                 contrast_prob=0.5,
                 saturation_range=0.9,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当label为空时，返回的tuple为(im, )，对应图像np.ndarray数据；
                   当label不为空时，返回的tuple为(im, label)，分别对应图像np.ndarray数据、图像类别id。
        """
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        ops = [brightness, contrast, saturation, hue]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob,
        }
        for id in range(len(ops)):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        im = im.astype('float32')
        if label is None:
            return (im, )
        else:
            return (im, label)


class ArrangeClassifier(ClsTransform):
    """获取训练/验证/预测所需信息。注意：此操作不需用户自己显示调用

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内。
    """

    def __init__(self, mode=None):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode must be in ['train', 'eval', 'test', 'quant']!")
        self.mode = mode

    def __call__(self, im, label=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            label (int): 每张图像所对应的类别序号。

        Returns:
            tuple: 当mode为'train'或'eval'时，返回(im, label)，分别对应图像np.ndarray数据、
                图像类别id；当mode为'test'或'quant'时，返回(im, )，对应图像np.ndarray数据。
        """
        im = permute(im, False).astype('float32')
        if self.mode == 'train' or self.mode == 'eval':
            outputs = (im, label)
        else:
            outputs = (im, )
        return outputs


class ComposedClsTransforms(Compose):
    """ 分类模型的基础Transforms流程，具体如下
        训练阶段：
        1. 随机从图像中crop一块子图，并resize成crop_size大小
        2. 将1的输出按0.5的概率随机进行水平翻转
        3. 将图像进行归一化
        验证/预测阶段：
        1. 将图像按比例Resize，使得最小边长度为crop_size[0] * 1.14
        2. 从图像中心crop出一个大小为crop_size的图像
        3. 将图像进行归一化

        Args:
            mode(str): 图像处理流程所处阶段，训练/验证/预测，分别对应'train', 'eval', 'test'
            crop_size(int|list): 输入模型里的图像大小
            mean(list): 图像均值
            std(list): 图像方差
            random_horizontal_flip(bool): 是否以0.5的概率使用随机水平翻转增强，该仅在mode为`train`时生效，默认为True
    """

    def __init__(self,
                 mode,
                 crop_size=[224, 224],
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 random_horizontal_flip=True):
        width = crop_size
        if isinstance(crop_size, list):
            if crop_size[0] != crop_size[1]:
                raise Exception(
                    "In classifier model, width and height should be equal, please modify your parameter `crop_size`"
                )
            width = crop_size[0]
        if width % 32 != 0:
            raise Exception(
                "In classifier model, width and height should be multiple of 32, e.g 224、256、320...., please modify your parameter `crop_size`"
            )

        if mode == 'train':
            # 训练时的transforms，包含数据增强
            transforms = [
                RandomCrop(crop_size=width), Normalize(
                    mean=mean, std=std)
            ]
            if random_horizontal_flip:
                transforms.insert(0, RandomHorizontalFlip())
        else:
            # 验证/预测时的transforms
            transforms = [
                ResizeByShort(short_size=int(width * 1.14)),
                CenterCrop(crop_size=width), Normalize(
                    mean=mean, std=std)
            ]

        super(ComposedClsTransforms, self).__init__(transforms)
