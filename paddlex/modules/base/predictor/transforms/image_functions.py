# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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



import cv2


def resize(im, target_size, interp):
    """ resize image to target size """
    w, h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def flip_h(im):
    """ flip image horizontally """
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    return im


def flip_v(im):
    """ flip image vertically """
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    return im


def slice(im, coords):
    """ slice the image """
    x1, y1, x2, y2 = coords
    im = im[y1:y2, x1:x2, ...]
    return im


def pad(im, pad, val):
    """ padding image by value """
    if isinstance(pad, int):
        pad = [pad] * 4
    if len(pad) != 4:
        raise ValueError
    chns = 1 if im.ndim == 2 else im.shape[2]
    im = cv2.copyMakeBorder(im, *pad, cv2.BORDER_CONSTANT, value=(val, ) * chns)
    return im
