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
    transforms for classification in PaddleX<2.0
"""

import math
import numpy as np
import cv2
from PIL import Image
from .operators import Transform, Compose, RandomHorizontalFlip, RandomVerticalFlip, Normalize, \
    ResizeByShort, CenterCrop, RandomDistort, ArrangeClassifier


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
        super(RandomCrop, self).__init__()
        self.crop_size = crop_size
        self.lower_scale = lower_scale
        self.lower_ratio = lower_ratio
        self.upper_ratio = upper_ratio

    def apply_im(self, image):
        scale = [self.lower_scale, 1.0]
        ratio = [self.lower_ratio, self.upper_ratio]
        aspect_ratio = math.sqrt(np.random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio
        bound = min((float(image.shape[0]) / image.shape[1]) / (h**2),
                    (float(image.shape[1]) / image.shape[0]) / (w**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)
        target_area = image.shape[0] * image.shape[1] * np.random.uniform(
            scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)
        i = np.random.randint(0, image.shape[0] - h + 1)
        j = np.random.randint(0, image.shape[1] - w + 1)
        image = image[i:i + h, j:j + w, :]
        image = cv2.resize(image, (self.crop_size, self.crop_size))
        return image

    def apply(self, sample):
        sample['image'] = self.apply_im(sample['image'])
        return sample


class RandomRotate(Transform):
    def __init__(self, rotate_range=30, prob=.5):
        """
        Randomly rotate image(s) by an arbitrary angle between -rotate_range and rotate_range.
        Args:
            rotate_range(int, optional): Range of the rotation angle. Defaults to 30.
            prob(float, optional): Probability of operating rotation. Defaults to .5.
        """
        self.rotate_range = rotate_range
        self.prob = prob

    def apply_im(self, image, angle):
        image = image.astype('uint8')
        image = Image.fromarray(image)
        image = image.rotate(angle)
        image = np.asarray(image).astype('float32')
        return image

    def apply(self, sample):
        rotate_lower = -self.rotate_range
        rotate_upper = self.rotate_range

        if np.random.uniform(0, 1) < self.prob:
            angle = np.random.uniform(rotate_lower, rotate_upper)
            sample['image'] = self.apply_im(sample['image'], angle)

        return sample


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
