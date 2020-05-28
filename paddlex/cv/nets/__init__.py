# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from .resnet import ResNet
from .darknet import DarkNet
from .detection import FasterRCNN
from .mobilenet_v1 import MobileNetV1
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .segmentation import UNet
from .segmentation import DeepLabv3p
from .xception import Xception
from .densenet import DenseNet
from .shufflenet_v2 import ShuffleNetV2
from .hrnet import HRNet


def resnet18(input, num_classes=1000):
    model = ResNet(layers=18, num_classes=num_classes)
    return model(input)


def resnet34(input, num_classes=1000):
    model = ResNet(layers=34, num_classes=num_classes)
    return model(input)


def resnet50(input, num_classes=1000):
    model = ResNet(layers=50, num_classes=num_classes)
    return model(input)


def resnet101(input, num_classes=1000):
    model = ResNet(layers=101, num_classes=num_classes)
    return model(input)


def resnet50_vd(input, num_classes=1000):
    model = ResNet(layers=50, num_classes=num_classes, variant='d')
    return model(input)


def resnet50_vd_ssld(input, num_classes=1000):
    model = ResNet(
        layers=50,
        num_classes=num_classes,
        variant='d',
        lr_mult_list=[1.0, 0.1, 0.2, 0.2, 0.3])
    return model(input)


def resnet101_vd_ssld(input, num_classes=1000):
    model = ResNet(
        layers=101,
        num_classes=num_classes,
        variant='d',
        lr_mult_list=[1.0, 0.1, 0.2, 0.2, 0.3])
    return model(input)


def resnet101_vd(input, num_classes=1000):
    model = ResNet(layers=101, num_classes=num_classes, variant='d')
    return model(input)


def darknet53(input, num_classes=1000):
    model = DarkNet(depth=53, num_classes=num_classes, bn_act='relu')
    return model(input)


def mobilenetv1(input, num_classes=1000):
    model = MobileNetV1(num_classes=num_classes)
    return model(input)


def mobilenetv2(input, num_classes=1000):
    model = MobileNetV2(num_classes=num_classes)
    return model(input)


def mobilenetv3_small(input, num_classes=1000):
    model = MobileNetV3(num_classes=num_classes, model_name='small')
    return model(input)


def mobilenetv3_large(input, num_classes=1000):
    model = MobileNetV3(num_classes=num_classes, model_name='large')
    return model(input)


def mobilenetv3_small_ssld(input, num_classes=1000):
    model = MobileNetV3(
        num_classes=num_classes,
        model_name='small',
        lr_mult_list=[0.25, 0.25, 0.5, 0.5, 0.75])
    return model(input)


def mobilenetv3_large_ssld(input, num_classes=1000):
    model = MobileNetV3(
        num_classes=num_classes,
        model_name='large',
        lr_mult_list=[0.25, 0.25, 0.5, 0.5, 0.75])
    return model(input)


def xception65(input, num_classes=1000):
    model = Xception(layers=65, num_classes=num_classes)
    return model(input)


def xception71(input, num_classes=1000):
    model = Xception(layers=71, num_classes=num_classes)
    return model(input)


def xception41(input, num_classes=1000):
    model = Xception(layers=41, num_classes=num_classes)
    return model(input)


def densenet121(input, num_classes=1000):
    model = DenseNet(layers=121, num_classes=num_classes)
    return model(input)


def densenet161(input, num_classes=1000):
    model = DenseNet(layers=161, num_classes=num_classes)
    return model(input)


def densenet201(input, num_classes=1000):
    model = DenseNet(layers=201, num_classes=num_classes)
    return model(input)


def shufflenetv2(input, num_classes=1000):
    model = ShuffleNetV2(num_classes=num_classes)
    return model(input)


def hrnet_w18(input, num_classes=1000):
    model = HRNet(width=18, num_classes=num_classes)
    return model(input)
