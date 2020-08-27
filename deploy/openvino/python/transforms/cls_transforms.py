# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from .ops import *
import random
import os.path as osp
import numpy as np
from PIL import Image, ImageEnhance


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
                    "im should be 3-dimension, but now is {}-dimensions".
                    format(len(im.shape)))
        else:
            try:
                im = cv2.imread(im).astype('float32')
            except:
                raise TypeError('Can\'t read The image file {}!'.format(im))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        for op in self.transforms:
            outputs = op(im, label)
            im = outputs[0]
            if len(outputs) == 2:
                label = outputs[1]
        return outputs

    def add_augmenters(self, augmenters):
        if not isinstance(augmenters, list):
            raise Exception(
                "augmenters should be list type in func add_augmenters()")
        transform_names = [type(x).__name__ for x in self.transforms]
        for aug in augmenters:
            if type(aug).__name__ in transform_names:
                print(
                    "{} is already in ComposedTransforms, need to remove it from add_augmenters().".
                    format(type(aug).__name__))
        self.transforms = augmenters + self.transforms


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
        if self.max_size > 0 and np.round(scale *
                                          im_long_size) > self.max_size:
            scale = float(self.max_size) / float(im_long_size)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im = cv2.resize(
            im, (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR)

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
    """

    def __init__(self,
                 mode,
                 crop_size=[224, 224],
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
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
            pass
        else:
            # 验证/预测时的transforms
            transforms = [
                ResizeByShort(short_size=int(width * 1.14)),
                CenterCrop(crop_size=width), Normalize(
                    mean=mean, std=std)
            ]

        super(ComposedClsTransforms, self).__init__(transforms)
