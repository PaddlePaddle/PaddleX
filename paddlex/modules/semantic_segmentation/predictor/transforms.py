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

import numpy as np
from PIL import Image

from ....utils import logging
from ...base import BaseTransform
from ...base.predictor.io.writers import ImageWriter
from .keys import SegKeys as K

__all__ = ['GeneratePCMap', 'SaveSegResults']


class GeneratePCMap(BaseTransform):
    """ GeneratePCMap """

    def __init__(self, color_map=None):
        super().__init__()
        self.color_map = color_map

    def apply(self, data):
        """ apply """
        pred = data[K.SEG_MAP]
        pc_map = self.get_pseudo_color_map(pred)
        data[K.PC_MAP] = pc_map
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.SEG_MAP]

    @classmethod
    def get_output_keys(cls):
        """ get input keys """
        return [K.PC_MAP]

    def get_pseudo_color_map(self, pred):
        """ get_pseudo_color_map """
        if pred.min() < 0 or pred.max() > 255:
            raise ValueError("`pred` cannot be cast to uint8.")
        pred = pred.astype(np.uint8)
        pred_mask = Image.fromarray(pred, mode='P')
        if self.color_map is None:
            color_map = self._get_color_map_list(256)
        else:
            color_map = self.color_map
        pred_mask.putpalette(color_map)
        return pred_mask

    @staticmethod
    def _get_color_map_list(num_classes, custom_color=None):
        """ _get_color_map_list """
        num_classes += 1
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        color_map = color_map[3:]

        if custom_color:
            color_map[:len(custom_color)] = custom_color
        return color_map


class SaveSegResults(BaseTransform):
    """ SaveSegResults """
    _PC_MAP_SUFFIX = '_pc'
    _FILE_EXT = '.png'

    def __init__(self, save_dir, save_pc_map=True):
        super().__init__()
        self.save_dir = save_dir
        self.save_pc_map = save_pc_map

        # We use pillow backend to save both numpy arrays and PIL Image objects
        self._writer = ImageWriter(backend='pillow')

    def apply(self, data):
        """ apply """
        seg_map = data[K.SEG_MAP]
        ori_path = data[K.IM_PATH]
        file_name = os.path.basename(ori_path)
        file_name = self._replace_ext(file_name, self._FILE_EXT)
        seg_map_save_path = os.path.join(self.save_dir, file_name)
        self._write_im(seg_map_save_path, seg_map)
        if self.save_pc_map:
            if K.PC_MAP in data:
                pc_map_save_path = self._add_suffix(seg_map_save_path,
                                                    self._PC_MAP_SUFFIX)
                pc_map = data[K.PC_MAP]
                self._write_im(pc_map_save_path, pc_map)
            else:
                logging.warning(f"The {K.PC_MAP} result don't exist!")
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IM_PATH, K.SEG_MAP]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []

    def _write_im(self, path, im):
        """ write image """
        if os.path.exists(path):
            logging.warning(f"{path} already exists. Overwriting it.")
        self._writer.write(path, im)

    @staticmethod
    def _add_suffix(path, suffix):
        """ add suffix """
        stem, ext = os.path.splitext(path)
        return stem + suffix + ext

    @staticmethod
    def _replace_ext(path, new_ext):
        """ replace ext """
        stem, _ = os.path.splitext(path)
        return stem + new_ext


class PrintResult(BaseTransform):
    """ Print Result Transform """

    def apply(self, data):
        """ apply """
        logging.info("The prediction result is:")
        logging.info(f"keys: {data.keys()}")
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.SEG_MAP]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []
