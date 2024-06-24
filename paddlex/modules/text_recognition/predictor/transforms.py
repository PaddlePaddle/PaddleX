# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2024 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Author: PaddlePaddle Authors
"""
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
from .keys import TextRecKeys as K

__all__ = ['OCRReisizeNormImg', 'CTCLabelDecode', 'SaveTextRecResults']


class OCRReisizeNormImg(BaseTransform):
    """ for ocr image resize and normalization """

    def __init__(self, rec_image_shape=[3, 48, 320]):
        super().__init__()
        self.rec_image_shape = rec_image_shape

    def resize_norm_img(self, img, max_wh_ratio):
        """ resize and normalize the img """
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def apply(self, data):
        """ apply """
        imgC, imgH, imgW = self.rec_image_shape
        max_wh_ratio = imgW / imgH
        w, h = data[K.ORI_IM_SIZE]
        wh_ratio = w * 1.0 / h
        max_wh_ratio = max(max_wh_ratio, wh_ratio)
        data[K.IMAGE] = self.resize_norm_img(data[K.IMAGE], max_wh_ratio)
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IMAGE, K.ORI_IM_SIZE]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.IMAGE]


class BaseRecLabelDecode(BaseTransform):
    """ Convert between text-label and text-index """

    def __init__(self, character_str=None):
        self.reverse = False
        dict_character = character_str if character_str is not None else "0123456789abcdefghijklmnopqrstuvwxyz"

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        """ pred_reverse """
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        """ add_special_char """
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        """ get_ignored_tokens """
        return [0]  # for ctc blank

    def apply(self, data):
        """ apply """
        preds = data[K.REC_PROBS]
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        data[K.REC_TEXT] = []
        data[K.REC_SCORE] = []
        for t in text:
            data[K.REC_TEXT].append(t[0])
            data[K.REC_SCORE].append(t[1])
        return data

    @classmethod
    def get_input_keys(cls):
        """ get_input_keys """
        return [K.REC_PROBS]

    @classmethod
    def get_output_keys(cls):
        """ get_output_keys """
        return [K.REC_TEXT, K.REC_SCORE]


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(self, post_process_cfg=None):
        assert post_process_cfg['name'] == 'CTCLabelDecode'
        character_str = post_process_cfg['character_dict']
        super().__init__(character_str)

    def apply(self, data):
        """ apply """
        preds = data[K.REC_PROBS]
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        data[K.REC_TEXT] = []
        data[K.REC_SCORE] = []
        for t in text:
            data[K.REC_TEXT].append(t[0])
            data[K.REC_SCORE].append(t[1])
        return data

    def add_special_char(self, dict_character):
        """ add_special_char """
        dict_character = ['blank'] + dict_character
        return dict_character

    @classmethod
    def get_input_keys(cls):
        """ get_input_keys """
        return [K.REC_PROBS]

    @classmethod
    def get_output_keys(cls):
        """ get_output_keys """
        return [K.REC_TEXT, K.REC_SCORE]


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
        for text, score in zip(data[K.REC_TEXT], data[K.REC_SCORE]):
            line = text + '\t' + str(score) + '\n'
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
