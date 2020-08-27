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

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

import random
import os.path as osp
import numpy as np

import cv2
from PIL import Image, ImageEnhance

from .ops import *
from .box_utils import *


class DetTransform:
    """检测数据处理基类
    """

    def __init__(self):
        pass


class Compose(DetTransform):
    """根据数据预处理/增强列表对输入数据进行操作。
       所有操作的输入图像流形状均是[H, W, C]，其中H为图像高，W为图像宽，C为图像通道数。

    Args:
        transforms (list): 数据预处理/增强列表。

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
        self.use_mixup = False
        for t in self.transforms:
            if type(t).__name__ == 'MixupImage':
                self.use_mixup = True

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息，dict中的字段如下：
                - im_id (np.ndarray): 图像序列号，形状为(1,)。
                - image_shape (np.ndarray): 图像原始大小，形状为(2,)，
                                        image_shape[0]为高，image_shape[1]为宽。
                - mixup (list): list为[im, im_info, label_info]，分别对应
                                与当前图像进行mixup的图像np.ndarray数据、图像相关信息、标注框相关信息；
                                注意，当前epoch若无需进行mixup，则无该字段。
            label_info (dict): 存储与标注框相关的信息，dict中的字段如下：
                - gt_bbox (np.ndarray): 真实标注框坐标[x1, y1, x2, y2]，形状为(n, 4)，
                                   其中n代表真实标注框的个数。
                - gt_class (np.ndarray): 每个真实标注框对应的类别序号，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - gt_score (np.ndarray): 每个真实标注框对应的混合得分，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - gt_poly (list): 每个真实标注框内的多边形分割区域，每个分割区域由点的x、y坐标组成，
                                  长度为n，其中n代表真实标注框的个数。
                - is_crowd (np.ndarray): 每个真实标注框中是否是一组对象，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - difficult (np.ndarray): 每个真实标注框中的对象是否为难识别对象，形状为(n, 1)，
                                     其中n代表真实标注框的个数。
        Returns:
            tuple: 根据网络所需字段所组成的tuple；
                字段由transforms中的最后一个数据预处理操作决定。
        """

        def decode_image(im_file, im_info, label_info):
            if im_info is None:
                im_info = dict()
            if isinstance(im_file, np.ndarray):
                if len(im_file.shape) != 3:
                    raise Exception(
                        "im should be 3-dimensions, but now is {}-dimensions".
                        format(len(im_file.shape)))
                im = im_file
            else:
                try:
                    im = cv2.imread(im_file).astype('float32')
                except:
                    raise TypeError('Can\'t read The image file {}!'.format(
                        im_file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # make default im_info with [h, w, 1]
            im_info['im_resize_info'] = np.array(
                [im.shape[0], im.shape[1], 1.], dtype=np.float32)
            im_info['image_shape'] = np.array([im.shape[0],
                                               im.shape[1]]).astype('int32')
            if not self.use_mixup:
                if 'mixup' in im_info:
                    del im_info['mixup']
            # decode mixup image
            if 'mixup' in im_info:
                im_info['mixup'] = \
                  decode_image(im_info['mixup'][0],
                               im_info['mixup'][1],
                               im_info['mixup'][2])
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)

        outputs = decode_image(im, im_info, label_info)
        im = outputs[0]
        im_info = outputs[1]
        if len(outputs) == 3:
            label_info = outputs[2]
        for op in self.transforms:
            if im is None:
                return None
            outputs = op(im, im_info, label_info)
            im = outputs[0]
        return outputs

    def add_augmenters(self, augmenters):
        if not isinstance(augmenters, list):
            raise Exception(
                "augmenters should be list type in func add_augmenters()")
        transform_names = [type(x).__name__ for x in self.transforms]
        for aug in augmenters:
            if type(aug).__name__ in transform_names:
                print("{} is already in ComposedTransforms, need to remove it from add_augmenters().".format(type(aug).__name__))
        self.transforms = augmenters + self.transforms


class ResizeByShort(DetTransform):
    """根据图像的短边调整图像大小（resize）。

    1. 获取图像的长边和短边长度。
    2. 根据短边与short_size的比例，计算长边的目标长度，
       此时高、宽的resize比例为short_size/原图短边长度。
    3. 如果max_size>0，调整resize比例：
       如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度。
    4. 根据调整大小的比例对图像进行resize。

    Args:
        target_size (int): 短边目标长度。默认为800。
        max_size (int): 长边目标长度的最大限制。默认为1333。

     Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, short_size=800, max_size=1333):
        self.max_size = int(max_size)
        if not isinstance(short_size, int):
            raise TypeError(
                "Type of short_size is invalid. Must be Integer, now is {}".
                format(type(short_size)))
        self.short_size = short_size
        if not (isinstance(self.max_size, int)):
            raise TypeError("max_size: input type is invalid.")

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，im_info更新字段为：
                       - im_resize_info (np.ndarray): resize后的图像高、resize后的图像宽、resize后的图像相对原始图的缩放比例
                                                 三者组成的np.ndarray，形状为(3,)。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeByShort: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('ResizeByShort: image is not 3-dimensional.')
        im_short_size = min(im.shape[0], im.shape[1])
        im_long_size = max(im.shape[0], im.shape[1])
        scale = float(self.short_size) / im_short_size
        if self.max_size > 0 and np.round(scale *
                                          im_long_size) > self.max_size:
            scale = float(self.max_size) / float(im_long_size)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im_resize_info = [resized_height, resized_width, scale]
        im = cv2.resize(
            im, (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR)
        im_info['im_resize_info'] = np.array(im_resize_info).astype(np.float32)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class Padding(DetTransform):
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
        self.coarsest_stride = coarsest_stride
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
        self.target_size = target_size

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
            ValueError: coarsest_stride，target_size需有且只有一个被指定。
            ValueError: target_size小于原图的大小。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("Padding: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Padding: image is not 3-dimensional.')
        im_h, im_w, im_c = im.shape[:]

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
        padding_im = np.zeros(
            (padding_im_h, padding_im_w, im_c), dtype=np.float32)
        padding_im[:im_h, :im_w, :] = im
        if label_info is None:
            return (padding_im, im_info)
        else:
            return (padding_im, im_info, label_info)


class Resize(DetTransform):
    """调整图像大小（resize）。

    - 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
    - 当目标大小（target_size）类型为list或tuple时，根据插值方式，
      将图像resize为target_size。
    注意：当插值方式为“RANDOM”时，则随机选取一种插值方式进行resize。

    Args:
        target_size (int/list/tuple): 短边目标长度。默认为608。
        interp (str): resize的插值方式，与opencv的插值方式对应，取值范围为
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为"LINEAR"。

    Raises:
        TypeError: 形参数据类型不满足需求。
        ValueError: 插值方式不在['NEAREST', 'LINEAR', 'CUBIC',
                    'AREA', 'LANCZOS4', 'RANDOM']中。
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=608, interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("interp should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise TypeError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)




class Normalize(DetTransform):
    """对图像进行标准化。

    1. 归一化图像到到区间[0.0, 1.0]。
    2. 对图像进行减均值除以标准差操作。

    Args:
        mean (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。
        std (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。

    Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise TypeError("NormalizeImage: input type is invalid.")
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise TypeError('NormalizeImage: std is invalid!')

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
        """
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)




class ArrangeYOLOv3(DetTransform):
    """获取YOLOv3模型训练/验证/预测所需信息。

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

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当mode为'train'时，返回(im, gt_bbox, gt_class, gt_score, im_shape)，分别对应
                图像np.ndarray数据、真实标注框、真实标注框对应的类别、真实标注框混合得分、图像大小信息；
                当mode为'eval'时，返回(im, im_shape, im_id, gt_bbox, gt_class, difficult)，
                分别对应图像np.ndarray数据、图像大小信息、图像id、真实标注框、真实标注框对应的类别、
                真实标注框是否为难识别对象；当mode为'test'或'quant'时，返回(im, im_shape)，
                分别对应图像np.ndarray数据、图像大小信息。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        im = permute(im, False)
        if self.mode == 'train':
            pass
        elif self.mode == 'eval':
            pass
        else:
            if im_info is None:
                raise TypeError('Cannot do ArrangeYolov3! ' +
                                'Becasuse the im_info can not be None!')
            im_shape = im_info['image_shape']
            outputs = (im, im_shape)
        return outputs




class ComposedYOLOv3Transforms(Compose):
    """YOLOv3模型的图像预处理流程，具体如下，
        训练阶段：
        1. 在前mixup_epoch轮迭代中，使用MixupImage策略，见https://paddlex.readthedocs.io/zh_CN/latest/apis/transforms/det_transforms.html#mixupimage
        2. 对图像进行随机扰动，包括亮度，对比度，饱和度和色调
        3. 随机扩充图像，见https://paddlex.readthedocs.io/zh_CN/latest/apis/transforms/det_transforms.html#randomexpand
        4. 随机裁剪图像
        5. 将4步骤的输出图像Resize成shape参数的大小
        6. 随机0.5的概率水平翻转图像
        7. 图像归一化
        验证/预测阶段：
        1. 将图像Resize成shape参数大小
        2. 图像归一化

        Args:
            mode(str): 图像处理流程所处阶段，训练/验证/预测，分别对应'train', 'eval', 'test'
            shape(list): 输入模型中图像的大小，输入模型的图像会被Resize成此大小
            mixup_epoch(int): 模型训练过程中，前mixup_epoch会使用mixup策略
            mean(list): 图像均值
            std(list): 图像方差
    """

    def __init__(self,
                 mode,
                 shape=[608, 608],
                 mixup_epoch=250,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        width = shape
        if isinstance(shape, list):
            if shape[0] != shape[1]:
                raise Exception(
                    "In YOLOv3 model, width and height should be equal")
            width = shape[0]
        if width % 32 != 0:
            raise Exception(
                "In YOLOv3 model, width and height should be multiple of 32, e.g 224、256、320...."
            )

        if mode == 'train':
            # 训练时的transforms，包含数据增强
            pass
        else:
            # 验证/预测时的transforms
            transforms = [
                Resize(
                    target_size=width, interp='CUBIC'), Normalize(
                        mean=mean, std=std)
            ]
        super(ComposedYOLOv3Transforms, self).__init__(transforms)
