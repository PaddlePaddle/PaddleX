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

from . import cv
from paddlex.cv.transforms import cls_transforms
import paddlex.utils.logging as logging

transforms = cls_transforms


class ResNet18(cv.models.ResNet18):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet18, self).__init__(num_classes=num_classes)


class ResNet34(cv.models.ResNet34):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet34, self).__init__(num_classes=num_classes)


class ResNet50(cv.models.ResNet50):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet50, self).__init__(num_classes=num_classes)


class ResNet101(cv.models.ResNet101):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet101, self).__init__(num_classes=num_classes)


class ResNet50_vd(cv.models.ResNet50_vd):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet50_vd, self).__init__(num_classes=num_classes)


class ResNet101_vd(cv.models.ResNet101_vd):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet101_vd, self).__init__(num_classes=num_classes)


class ResNet50_vd_ssld(cv.models.ResNet50_vd_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet50_vd_ssld, self).__init__(num_classes=num_classes)


class ResNet101_vd_ssld(cv.models.ResNet101_vd_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ResNet101_vd_ssld, self).__init__(num_classes=num_classes)


class DarkNet53(cv.models.DarkNet53):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DarkNet53, self).__init__(num_classes=num_classes)


class MobileNetV1(cv.models.MobileNetV1):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV1, self).__init__(num_classes=num_classes)


class MobileNetV2(cv.models.MobileNetV2):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV2, self).__init__(num_classes=num_classes)


class MobileNetV3_small(cv.models.MobileNetV3_small):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_small, self).__init__(num_classes=num_classes)


class MobileNetV3_large(cv.models.MobileNetV3_large):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_large, self).__init__(num_classes=num_classes)


class MobileNetV3_small_ssld(cv.models.MobileNetV3_small_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_small_ssld, self).__init__(num_classes=num_classes)


class MobileNetV3_large_ssld(cv.models.MobileNetV3_large_ssld):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(MobileNetV3_large_ssld, self).__init__(num_classes=num_classes)


class Xception41(cv.models.Xception41):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(Xception41, self).__init__(num_classes=num_classes)


class Xception65(cv.models.Xception65):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(Xception65, self).__init__(num_classes=num_classes)


class DenseNet121(cv.models.DenseNet121):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DenseNet121, self).__init__(num_classes=num_classes)


class DenseNet161(cv.models.DenseNet161):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DenseNet161, self).__init__(num_classes=num_classes)


class DenseNet201(cv.models.DenseNet201):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(DenseNet201, self).__init__(num_classes=num_classes)


class ShuffleNetV2(cv.models.ShuffleNetV2):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(ShuffleNetV2, self).__init__(num_classes=num_classes)


class HRNet_W18(cv.models.HRNet_W18_C):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(HRNet_W18, self).__init__(num_classes=num_classes)


class AlexNet(cv.models.AlexNet):
    def __init__(self, num_classes=1000, input_channel=None):
        if input_channel is not None:
            logging.warning(
                "`input_channel` is deprecated in PaddleX 2.0 and won't take effect. Defaults to 3."
            )
        super(AlexNet, self).__init__(num_classes=num_classes)
