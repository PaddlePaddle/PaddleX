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

import numpy as np
from PIL import Image
import cv2
import paddle

from ....utils import logging
from ...base import BaseTransform
from ...base.predictor.io.writers import ImageWriter
from .keys import TableRecKeys as K

__all__ = ['TableLabelDecode', 'TableMasterLabelDecode', 'SaveTableResults']


class TableLabelDecode(BaseTransform):
    """ decode the table model outputs(probs) to character str"""

    def __init__(self,
                 character_dict_type='TableAttn_ch',
                 merge_no_span_structure=True):
        dict_character = []
        supported_dict = ['TableAttn_ch', 'TableAttn_en', 'TableMaster']
        if character_dict_type == 'TableAttn_ch':
            character_dict_name = 'table_structure_dict_ch.txt'
        elif character_dict_type == 'TableAttn_en':
            character_dict_name = 'table_structure_dict.txt'
        elif character_dict_type == 'TableMaster':
            character_dict_name = 'table_master_structure_dict.txt'
        else:
            assert False, " character_dict_type must in %s " \
                          % supported_dict
        character_dict_path = osp.abspath(
            osp.join(osp.dirname(__file__), character_dict_name))
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                dict_character.append(line)

        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        self.td_token = ['<td>', '<td', '<td></td>']

    def add_special_char(self, dict_character):
        """ add_special_char """
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        """ get_ignored_tokens """
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        """ get_beg_end_flag_idx """
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" \
                          % beg_or_end
        return idx

    def apply(self, data):
        """ apply """
        shape_list = data[K.SHAPE_LIST]

        structure_probs = data[K.STRUCTURE_PROB]
        bbox_preds = data[K.LOC_PROB]
        if isinstance(structure_probs, paddle.Tensor):
            structure_probs = structure_probs.numpy()
        if isinstance(bbox_preds, paddle.Tensor):
            bbox_preds = bbox_preds.numpy()
        post_result = self.decode(structure_probs, bbox_preds, shape_list)
        structure_str_list = post_result['structure_batch_list'][0]
        bbox_list = post_result['bbox_batch_list'][0]
        structure_str_list = structure_str_list[0]
        structure_str_list = [
            '<html>', '<body>', '<table>'
        ] + structure_str_list + ['</table>', '</body>', '</html>']
        data[K.BBOX_RES] = bbox_list
        data[K.HTML_RES] = structure_str_list
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.STRUCTURE_PROB, K.LOC_PROB, K.SHAPE_LIST]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.BBOX_RES, K.HTML_RES]

    def decode(self, structure_probs, bbox_preds, shape_list):
        """convert text-label into text-index.
        """
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            score_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                text = self.character[char_idx]
                if text in self.td_token:
                    bbox = bbox_preds[batch_idx, idx]
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, np.mean(score_list)])
            bbox_batch_list.append(np.array(bbox_list))
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    def decode_label(self, batch):
        """convert text-label into text-index.
        """
        structure_idx = batch[1]
        gt_bbox_list = batch[2]
        shape_list = batch[-1]
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        for batch_idx in range(batch_size):
            structure_list = []
            bbox_list = []
            for idx in range(len(structure_idx[batch_idx])):
                char_idx = int(structure_idx[batch_idx][idx])
                if idx > 0 and char_idx == end_idx:
                    break
                if char_idx in ignored_tokens:
                    continue
                structure_list.append(self.character[char_idx])

                bbox = gt_bbox_list[batch_idx][idx]
                if bbox.sum() != 0:
                    bbox = self._bbox_decode(bbox, shape_list[batch_idx])
                    bbox_list.append(bbox)
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        result = {
            'bbox_batch_list': bbox_batch_list,
            'structure_batch_list': structure_batch_list,
        }
        return result

    def _bbox_decode(self, bbox, shape):
        w, h = shape[:2]
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox


class TableMasterLabelDecode(TableLabelDecode):
    """ decode the table model outputs(probs) to character str"""

    def __init__(self,
                 character_dict_type='TableMaster',
                 box_shape='pad',
                 merge_no_span_structure=True):
        super(TableMasterLabelDecode, self).__init__(character_dict_type,
                                                     merge_no_span_structure)
        self.box_shape = box_shape
        assert box_shape in [
            'ori', 'pad'
        ], 'The shape used for box normalization must be ori or pad'

    def add_special_char(self, dict_character):
        """ add_special_char """
        self.beg_str = '<SOS>'
        self.end_str = '<EOS>'
        self.unknown_str = '<UKN>'
        self.pad_str = '<PAD>'
        dict_character = dict_character
        dict_character = dict_character + [
            self.unknown_str, self.beg_str, self.end_str, self.pad_str
        ]
        return dict_character

    def get_ignored_tokens(self):
        """ get_ignored_tokens """
        pad_idx = self.dict[self.pad_str]
        start_idx = self.dict[self.beg_str]
        end_idx = self.dict[self.end_str]
        unknown_idx = self.dict[self.unknown_str]
        return [start_idx, end_idx, pad_idx, unknown_idx]

    def _bbox_decode(self, bbox, shape):
        """ _bbox_decode """
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        if self.box_shape == 'pad':
            h, w = pad_h, pad_w
        bbox[0::2] *= w
        bbox[1::2] *= h
        bbox[0::2] /= ratio_w
        bbox[1::2] /= ratio_h
        x, y, w, h = bbox
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        bbox = np.array([x1, y1, x2, y2])
        return bbox


class SaveTableResults(BaseTransform):
    """ SaveTableResults """
    _TABLE_RES_SUFFIX = '_bbox'
    _FILE_EXT = '.png'

    # _DEFAULT_FILE_NAME = 'table_res_out.png'

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir

        # We use pillow backend to save both numpy arrays and PIL Image objects
        self._writer = ImageWriter(backend='pillow')

    def apply(self, data):
        """ apply """
        ori_path = data[K.IM_PATH]
        bbox_res = data[K.BBOX_RES]
        file_name = os.path.basename(ori_path)
        file_name = self._replace_ext(file_name, self._FILE_EXT)
        table_res_save_path = os.path.join(self.save_dir, file_name)

        if len(bbox_res) > 0 and len(bbox_res[0]) == 4:
            vis_img = self.draw_rectangle(data[K.ORI_IM], bbox_res)
        else:
            vis_img = self.draw_bbox(data[K.ORI_IM], bbox_res)
        table_res_save_path = self._add_suffix(table_res_save_path,
                                               self._TABLE_RES_SUFFIX)
        self._write_im(table_res_save_path, vis_img)
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IM_PATH, K.ORI_IM, K.BBOX_RES]

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
        """ _add_suffix """
        stem, ext = os.path.splitext(path)
        return stem + suffix + ext

    @staticmethod
    def _replace_ext(path, new_ext):
        """ _replace_ext """
        stem, _ = os.path.splitext(path)
        return stem + new_ext

    def draw_rectangle(self, img_path, boxes):
        """ draw_rectangle """
        boxes = np.array(boxes)
        img = cv2.imread(img_path)
        img_show = img.copy()
        for box in boxes.astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(img_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img_show

    def draw_bbox(self, image, boxes):
        """ draw_bbox """
        for box in boxes:
            box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
            image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
        return image


class PrintResult(BaseTransform):
    """ Print Result Transform """

    def apply(self, data):
        """ apply """
        logging.info("The prediction result is:")
        logging.info(data[K.BOXES])
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.BOXES]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []
