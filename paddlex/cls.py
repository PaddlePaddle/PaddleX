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

from .cv.models import BaseClassifier


class ResNet18(BaseClassifier):
    def __init__(self, num_classes=1000, sync_bn=True):
        super(ResNet18, self).__init__(
            model_name='ResNet18', num_classes=num_classes, sync_bn=sync_bn)


class ResNet34(BaseClassifier):
    def __init__(self, num_classes=1000, sync_bn=True):
        super(ResNet34, self).__init__(
            model_name='ResNet34', num_classes=num_classes, sync_bn=sync_bn)


class ResNet50(BaseClassifier):
    def __init__(self, num_classes=1000, sync_bn=True):
        super(ResNet50, self).__init__(
            model_name='ResNet50', num_classes=num_classes, sync_bn=sync_bn)


class ResNet101(BaseClassifier):
    def __init__(self, num_classes=1000, sync_bn=True):
        super(ResNet101, self).__init__(
            model_name='ResNet101', num_classes=num_classes, sync_bn=sync_bn)


class ResNet152(BaseClassifier):
    def __init__(self, num_classes=1000, sync_bn=True):
        super(ResNet152, self).__init__(
            model_name='ResNet152', num_classes=num_classes, sync_bn=sync_bn)
