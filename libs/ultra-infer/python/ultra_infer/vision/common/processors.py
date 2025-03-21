# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import

from ... import c_lib_wrap as C


class Processor:
    def __init__(self):
        self.processor = None

    def __call__(self, mat):
        """call for processing input.

        :param mat: The input data FDMat or FDMatBatch.
        """
        self.processor(mat)


class ResizeByShort(Processor):
    def __init__(self, target_size: int, interp=1, use_scale=True, max_hw=[]):
        """Create a ResizeByShort operation with the given parameters.

        :param target_size: The target short size to resize the image
        :param interp: Optionally, the interpolation mode for resizing image
        :param use_scale: Optionally, whether to scale image
        :param max_hw: Max spatial size which is used by ResizeByShort
        """
        self.processor = C.vision.processors.ResizeByShort(
            target_size, interp, use_scale, max_hw
        )


class CenterCrop(Processor):
    def __init__(self, width, height):
        """Create a CenterCrop operation with the given parameters.

        :param width: Desired width of the cropped image
        :param height: Desired height of the cropped image
        """
        self.processor = C.vision.processors.CenterCrop(width, height)


class Pad(Processor):
    def __init__(self, top: int, bottom: int, left: int, right: int, value=[]):
        """Create a Pad operation with the given parameters.

        :param top: The top padding
        :param bottom: The bottom padding
        :param left: The left padding
        :param right: The right padding
        :param value: the value that is used to pad on the input image
        """
        self.processor = C.vision.processors.Pad(top, bottom, left, right, value)


class NormalizeAndPermute(Processor):
    def __init__(self, mean=[], std=[], is_scale=True, min=[], max=[], swap_rb=False):
        """Creae a Normalize and a Permute operation with the given parameters.

        :param mean: A list containing the mean of each channel
        :param std: A list containing the standard deviation of each channel
        :param is_scale: Specifies if the image are being scaled or not
        :param min: A list containing the minimum value of each channel
        :param max: A list containing the maximum value of each channel
        """
        self.processor = C.vision.processors.NormalizeAndPermute(
            mean, std, is_scale, min, max, swap_rb
        )


class Cast(Processor):
    def __init__(self, dtype="float"):
        """Creat a new cast opereaton with given dtype

        :param dtype: Target dtype of the output
        """
        self.processor = C.vision.processors.Cast(dtype)


class HWC2CHW(Processor):
    def __init__(self):
        """Creat a new hwc2chw processor with default dtype.

        :return An instance of processor `HWC2CHW`
        """
        self.processor = C.vision.processors.HWC2CHW()


class Normalize(Processor):
    def __init__(self, mean, std, is_scale=True, min=[], max=[], swap_rb=False):
        """Creat a new normalize opereator with given paremeters.

        :param mean: A list containing the mean of each channel
        :param std: A list containing the standard deviation of each channel
        :param is_scale: Specifies if the image are being scaled or not
        :param min: A list containing the minimum value of each channel
        :param max: A list containing the maximum value of each channel
        """
        self.processor = C.vision.processors.Normalize(
            mean, std, is_scale, min, max, swap_rb
        )


class PadToSize(Processor):
    def __init__(self, width, height, value=[]):
        """Create a new PadToSize opereator with given parameters.

        :param width: Desired width of the output image
        :param height: Desired height of the output image
        :param value: Values to pad with
        """
        self.processor = C.vision.processors.PadToSize(width, height, value)


class Resize(Processor):
    def __init__(
        self, width, height, scale_w=-1.0, scale_h=-1.0, interp=1, use_scale=False
    ):
        """Create a Resize operation with the given parameters.

        :param width: Desired width of the output image
        :param height: Desired height of the output image
        :param scale_w: Scales the width in x-direction
        :param scale_h: Scales the height in y-direction
        :param interp: Optionally, the interpolation mode for resizing image
        :param use_scale: Optionally, whether to scale image
        """
        self.processor = C.vision.processors.Resize(
            width, height, scale_w, scale_h, interp, use_scale
        )


class StridePad(Processor):
    def __init__(self, stride, value=[]):
        """Create a StridePad processor with given parameters.

        :param stride: Stride of the processor
        :param value: Values to pad with
        """
        self.processor = C.vision.processors.StridePad(stride, value)
