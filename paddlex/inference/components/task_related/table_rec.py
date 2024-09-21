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

import numpy as np

from ..base import BaseComponent

__all__ = ["TableLabelDecode", "TableMasterLabelDecode"]


class TableLabelDecode(BaseComponent):
    """decode the table model outputs(probs) to character str"""

    ENABLE_BATCH = True

    INPUT_KEYS = ["pred", "ori_img_size"]
    OUTPUT_KEYS = ["bbox", "structure"]
    DEAULT_INPUTS = {"pred": "pred", "ori_img_size": "ori_img_size"}
    DEAULT_OUTPUTS = {"bbox": "bbox", "structure": "structure"}

    def __init__(self, merge_no_span_structure=True, dict_character=[]):
        super().__init__()

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
        self.td_token = ["<td>", "<td", "<td></td>"]

    def add_special_char(self, dict_character):
        """add_special_char"""
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def get_ignored_tokens(self):
        """get_ignored_tokens"""
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        """get_beg_end_flag_idx"""
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "unsupported type %s in get_beg_end_flag_idx" % beg_or_end
        return idx

    def apply(self, pred, ori_img_size):
        """apply"""
        bbox_preds, structure_probs = [], []
        for bbox_pred, stru_prob in pred:
            bbox_preds.append(bbox_pred)
            structure_probs.append(stru_prob)
        bbox_preds = np.array(bbox_preds)
        structure_probs = np.array(structure_probs)

        bbox_list, structure_str_list = self.decode(
            structure_probs, bbox_preds, ori_img_size
        )
        structure_str_list = [
            (
                ["<html>", "<body>", "<table>"]
                + structure
                + ["</table>", "</body>", "</html>"]
            )
            for structure in structure_str_list
        ]

        return [
            {"bbox": bbox, "structure": structure}
            for bbox, structure in zip(bbox_list, structure_str_list)
        ]

    def decode(self, structure_probs, bbox_preds, shape_list):
        """convert text-label into text-index."""
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
                    bbox_list.append(bbox.tolist())
                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append([structure_list, float(np.mean(score_list))])
            bbox_batch_list.append(bbox_list)

        return bbox_batch_list, structure_batch_list

    def decode_label(self, batch):
        """convert text-label into text-index."""
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
                    bbox_list.append(bbox.tolist())
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        return bbox_batch_list, structure_batch_list

    def _bbox_decode(self, bbox, shape):
        w, h = shape[:2]
        bbox[0::2] *= w
        bbox[1::2] *= h
        return bbox


class TableMasterLabelDecode(TableLabelDecode):
    """decode the table model outputs(probs) to character str"""

    def __init__(
        self,
        character_dict_type="TableMaster",
        box_shape="pad",
        merge_no_span_structure=True,
    ):
        super(TableMasterLabelDecode, self).__init__(
            character_dict_type, merge_no_span_structure
        )
        self.box_shape = box_shape
        assert box_shape in [
            "ori",
            "pad",
        ], "The shape used for box normalization must be ori or pad"

    def add_special_char(self, dict_character):
        """add_special_char"""
        self.beg_str = "<SOS>"
        self.end_str = "<EOS>"
        self.unknown_str = "<UKN>"
        self.pad_str = "<PAD>"
        dict_character = dict_character
        dict_character = dict_character + [
            self.unknown_str,
            self.beg_str,
            self.end_str,
            self.pad_str,
        ]
        return dict_character

    def get_ignored_tokens(self):
        """get_ignored_tokens"""
        pad_idx = self.dict[self.pad_str]
        start_idx = self.dict[self.beg_str]
        end_idx = self.dict[self.end_str]
        unknown_idx = self.dict[self.unknown_str]
        return [start_idx, end_idx, pad_idx, unknown_idx]

    def _bbox_decode(self, bbox, shape):
        """_bbox_decode"""
        h, w, ratio_h, ratio_w, pad_h, pad_w = shape
        if self.box_shape == "pad":
            h, w = pad_h, pad_w
        bbox[0::2] *= w
        bbox[1::2] *= h
        bbox[0::2] /= ratio_w
        bbox[1::2] /= ratio_h
        x, y, w, h = bbox
        x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
        bbox = np.array([x1, y1, x2, y2])
        return bbox
