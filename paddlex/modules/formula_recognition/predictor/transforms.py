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


import os
import os.path as osp

import re
import numpy as np
from PIL import Image
import cv2
import math
import paddle

from ....utils import logging
from ...base.predictor import BaseTransform
from ...base.predictor.io.writers import TextWriter
from .keys import FormulaRecKeys as K

__all__ = ['OCRReisizeNormImg',  'LaTeXOCRDecode', 'SaveTextRecResults']


class OCRReisizeNormImg(BaseTransform):
    """ for ocr image resize and normalization """

    def __init__(self, rec_image_shape=[3, 48, 320]):
        super().__init__()
        self.rec_image_shape = rec_image_shape

    def pad_(self, img, divable=32):
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)
        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))
        padded = Image.new("L", dims, 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size_(
        self,
        img,
        max_dimensions,
        min_dimensions,
    ):
        if max_dimensions is not None:
            ratios = [a / b for a, b in zip(img.size, max_dimensions)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(tuple(size.astype(int)), Image.BILINEAR)
        if min_dimensions is not None:
            # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
            padded_size = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, min_dimensions)
            ]
            if padded_size != list(img.size):  # assert hypothesis
                padded_im = Image.new("L", padded_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def norm_img_latexocr(self, img):
        # CAN only predict gray scale image
        shape = (1, 1, 3)
        mean = [0.7931, 0.7931, 0.7931]
        std = [0.1738, 0.1738, 0.1738]
        scale = 255.0
        min_dimensions = [32, 32]
        max_dimensions = [672, 192]
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")

        im_h, im_w = img.shape[:2]
        if (
            min_dimensions[0] <= im_w <= max_dimensions[0]
            and min_dimensions[1] <= im_h <= max_dimensions[1]
        ):
            pass
        else:
            img = Image.fromarray(np.uint8(img))
            img = self.minmax_size_(self.pad_(img), max_dimensions, min_dimensions)
            img = np.array(img)
            im_h, im_w = img.shape[:2]
            img = np.dstack([img, img, img])
        img = (img.astype("float32") * scale - mean) / std
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img = img[:, :, np.newaxis].transpose(2, 0, 1)
        img = img.astype("float32")
        return img

    def apply(self, data):
        """ apply """
        data[K.IMAGE] = self.norm_img_latexocr(data[K.IMAGE])
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IMAGE, K.ORI_IM_SIZE]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.IMAGE]

class LaTeXOCRDecode(object):
    """Convert between latex-symbol and symbol-index"""

    def __init__(self, rec_char_dict_path, **kwargs):
        from tokenizers import Tokenizer as TokenizerFast

        super(LaTeXOCRDecode, self).__init__()
        self.tokenizer = TokenizerFast.from_file(rec_char_dict_path)

    def post_process(self, s):
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = [x[0].replace(" ", "") for x in re.findall(text_reg, s)]
        s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s

    def decode(self, tokens):
        if len(tokens.shape) == 1:
            tokens = tokens[None, :]
       
        dec = [self.tokenizer.decode(tok) for tok in tokens]
        dec_str_list = [
            "".join(detok.split(" "))
            .replace("Ġ", " ")
            .replace("[EOS]", "")
            .replace("[BOS]", "")
            .replace("[PAD]", "")
            .strip()
            for detok in dec
        ]
        return [str(self.post_process(dec_str)) for dec_str in dec_str_list]

    def __call__(self, data):
        preds = data[K.REC_PROBS]
        text = self.decode(preds)
        data[K.REC_TEXT] = text[0]
        return data


class SaveTextRecResults(BaseTransform):
    """ SaveTextRecResults """
    _TEXT_REC_RES_SUFFIX = '_text_rec'
    _FILE_EXT = '.txt'

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        # We use python backend to save text object
        self._writer = TextWriter(backend='python')

    def apply(self, data):
        """ apply """
        ori_path = data[K.IM_PATH]
        file_name = os.path.basename(ori_path)
        file_name = self._replace_ext(file_name, self._FILE_EXT)
        text_rec_res_save_path = os.path.join(self.save_dir, file_name)
        rec_res = ''
        for text in data[K.REC_TEXT]:
            line = text + '\n'
            rec_res += line
        text_rec_res_save_path = self._add_suffix(text_rec_res_save_path,
                                                  self._TEXT_REC_RES_SUFFIX)
        self._write_txt(text_rec_res_save_path, rec_res)
        return data

    @classmethod
    def get_input_keys(cls):
        """ get_input_keys """
        return [K.IM_PATH, K.REC_TEXT, K.REC_SCORE]

    @classmethod
    def get_output_keys(cls):
        """ get_output_keys """
        return []

    def _write_txt(self, path, txt_str):
        """ _write_txt """
        if os.path.exists(path):
            logging.warning(f"{path} already exists. Overwriting it.")
        self._writer.write(path, txt_str)

    @staticmethod
    def _add_suffix(path, suffix):
        """ _add_suffix """
        stem, ext = os.path.splitext(path)
        return stem + suffix + ext

    @staticmethod
    def _replace_ext(path, new_ext):
        """ _replace_ext """
        stem, _ = os.path.splitext(path)
        return stem + new_ext



class PrintResult(BaseTransform):
    """ Print Result Transform """

    def apply(self, data):
        """ apply """
        logging.info("The prediction result is:")
        logging.info(data[K.REC_TEXT])
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.REC_TEXT]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []
