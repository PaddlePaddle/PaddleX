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

from .autoaugment import ImageNetPolicy as RawImageNetPolicy
from .randaugment import RandAugment as RawRandAugment
from .cutout import Cutout

from .hide_and_seek import HideAndSeek
from .random_erasing import RandomErasing
from .grid import GridMask

from .operators import DecodeImage
from .operators import ResizeImage
from .operators import CropImage
from .operators import RandCropImage
from .operators import RandFlipImage
from .operators import NormalizeImage
from .operators import ToCHWImage

from .batch_operators import MixupOperator
from .batch_operators import CutmixOperator
from .batch_operators import FmixOperator

import six
import numpy as np
from PIL import Image


def transform(data, ops=[]):
    """ transform """
    for op in ops:
        data = op(data)
    return data


class AutoAugment(RawImageNetPolicy):
    """ ImageNetPolicy wrapper to auto fit different img types """

    def __init__(self, *args, **kwargs):
        if six.PY2:
            super(AutoAugment, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        if six.PY2:
            img = super(AutoAugment, self).__call__(img)
        else:
            img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img


class RandAugment(RawRandAugment):
    """ RandAugment wrapper to auto fit different img types """

    def __init__(self, *args, **kwargs):
        if six.PY2:
            super(RandAugment, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        if six.PY2:
            img = super(RandAugment, self).__call__(img)
        else:
            img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)

        return img
