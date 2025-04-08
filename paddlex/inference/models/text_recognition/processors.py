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


import math
import re
from typing import List

import numpy as np

from ....utils.deps import class_requires_deps, is_dep_available
from ...utils.benchmark import benchmark

if is_dep_available("opencv-contrib-python"):
    import cv2


@benchmark.timeit
@class_requires_deps("opencv-contrib-python")
class OCRReisizeNormImg:
    """for ocr image resize and normalization"""

    def __init__(self, rec_image_shape=[3, 48, 320], input_shape=None):
        super().__init__()
        self.rec_image_shape = rec_image_shape
        self.input_shape = input_shape
        self.max_imgW = 3200

    def resize_norm_img(self, img, max_wh_ratio):
        """resize and normalize the img"""
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if imgW > self.max_imgW:
            resized_image = cv2.resize(img, (self.max_imgW, imgH))
            resized_w = self.max_imgW
            imgW = self.max_imgW
        else:
            h, w = img.shape[:2]
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, imgs):
        """apply"""
        if self.input_shape is None:
            return [self.resize(img) for img in imgs]
        else:
            return [self.staticResize(img) for img in imgs]

    def resize(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = imgW / imgH
        h, w = img.shape[:2]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        img = self.resize_norm_img(img, max_wh_ratio)
        return img

    def staticResize(self, img):
        imgC, imgH, imgW = self.input_shape
        resized_image = cv2.resize(img, (int(imgW), int(imgH)))
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        return resized_image


@benchmark.timeit
class BaseRecLabelDecode:
    """Convert between text-label and text-index"""

    def __init__(self, character_str=None, use_space_char=True):
        super().__init__()
        self.reverse = False
        character_list = (
            list(character_str)
            if character_str is not None
            else list("0123456789abcdefghijklmnopqrstuvwxyz")
        )
        if use_space_char:
            character_list.append(" ")

        character_list = self.add_special_char(character_list)
        self.dict = {}
        for i, char in enumerate(character_list):
            self.dict[char] = i
        self.character = character_list

    def pred_reverse(self, pred):
        """pred_reverse"""
        pred_re = []
        c_current = ""
        for c in pred:
            if not bool(re.search("[a-zA-Z0-9 :*./%+-]", c)):
                if c_current != "":
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ""
            else:
                c_current += c
        if c_current != "":
            pred_re.append(c_current)

        return "".join(pred_re[::-1])

    def add_special_char(self, character_list):
        """add_special_char"""
        return character_list

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """convert text-index into text-label."""
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id] for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = "".join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        """get_ignored_tokens"""
        return [0]  # for ctc blank

    def __call__(self, pred):
        """apply"""
        preds = np.array(pred)
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])
        return texts, scores


@benchmark.timeit
class CTCLabelDecode(BaseRecLabelDecode):
    """Convert between text-label and text-index"""

    def __init__(self, character_list=None, use_space_char=True):
        super().__init__(character_list, use_space_char=use_space_char)

    def __call__(self, pred):
        """apply"""
        preds = np.array(pred[0])
        preds_idx = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=-1)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        texts = []
        scores = []
        for t in text:
            texts.append(t[0])
            scores.append(t[1])
        return texts, scores

    def add_special_char(self, character_list):
        """add_special_char"""
        character_list = ["blank"] + character_list
        return character_list


@benchmark.timeit
class ToBatch:
    """A class for batching and padding images to a uniform width."""

    def __pad_imgs(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Pad images to the maximum width in the batch.

        Args:
            imgs (list of np.ndarrays): List of images to pad.

        Returns:
            list of np.ndarrays: List of padded images.
        """
        max_width = max(img.shape[2] for img in imgs)
        padded_imgs = []
        for img in imgs:
            _, height, width = img.shape
            pad_width = max_width - width
            padded_img = np.pad(
                img,
                ((0, 0), (0, 0), (0, pad_width)),
                mode="constant",
                constant_values=0,
            )
            padded_imgs.append(padded_img)
        return padded_imgs

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Call method to pad images and stack them into a batch.

        Args:
            imgs (list of np.ndarrays): List of images to process.

        Returns:
            list of np.ndarrays: List containing a stacked tensor of the padded images.
        """
        imgs = self.__pad_imgs(imgs)
        return [np.stack(imgs, axis=0).astype(dtype=np.float32, copy=False)]
