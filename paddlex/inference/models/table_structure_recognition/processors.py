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


import numpy as np

from ...utils.benchmark import benchmark
from ..common.vision import funcs as F


@benchmark.timeit
class Pad:
    """Pad the image."""

    def __init__(self, target_size, val=127.5):
        """
        Initialize the instance.

        Args:
            target_size (list|tuple|int): Target width and height of the image after
                padding.
            val (float, optional): Value to fill the padded area. Default: 127.5.
        """
        super().__init__()

        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size

        self.val = val

    def apply(self, img):
        """apply"""
        h, w = img.shape[:2]
        tw, th = self.target_size
        ph = th - h
        pw = tw - w

        if ph < 0 or pw < 0:
            raise ValueError(
                f"Input image ({w}, {h}) smaller than the target size ({tw}, {th})."
            )
        else:
            img = F.pad(img, pad=(0, ph, 0, pw), val=self.val)

        return [img, [img.shape[1], img.shape[0]]]

    def __call__(self, imgs):
        """apply"""
        return [self.apply(img) for img in imgs]


@benchmark.timeit
class TableLabelDecode:
    """decode the table model outputs(probs) to character str"""

    ENABLE_BATCH = True

    INPUT_KEYS = ["pred", "img_size", "ori_img_size"]
    OUTPUT_KEYS = ["bbox", "structure", "structure_score"]
    DEAULT_INPUTS = {
        "pred": "pred",
        "img_size": "img_size",
        "ori_img_size": "ori_img_size",
    }
    DEAULT_OUTPUTS = {
        "bbox": "bbox",
        "structure": "structure",
        "structure_score": "structure_score",
    }

    def __init__(self, model_name, merge_no_span_structure=True, dict_character=[]):
        super().__init__()

        if merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")
        self.model_name = model_name

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

    def __call__(self, pred, img_size, ori_img_size):
        """apply"""
        bbox_preds = np.array([list(pred[0][0])])
        structure_probs = np.array([list(pred[1][0])])

        bbox_list, structure_str_list, structure_score = self.decode(
            structure_probs, bbox_preds, img_size, ori_img_size
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
            {"bbox": bbox, "structure": structure, "structure_score": structure_score}
            for bbox, structure in zip(bbox_list, structure_str_list)
        ]

    def decode(self, structure_probs, bbox_preds, padding_size, ori_img_size):
        """convert text-label into text-index."""
        ignored_tokens = self.get_ignored_tokens()
        end_idx = self.dict[self.end_str]

        structure_idx = structure_probs.argmax(axis=2)
        structure_probs = structure_probs.max(axis=2)

        structure_batch_list = []
        bbox_batch_list = []
        batch_size = len(structure_idx)
        bbox_list = []
        scale_list = []
        scales = [0] * 8
        for batch_idx in range(batch_size):
            structure_list = []
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
                    h_scale, w_scale = self._get_bbox_scales(
                        padding_size[batch_idx], ori_img_size[batch_idx]
                    )
                    scales[0::2] = [h_scale] * 4
                    scales[1::2] = [w_scale] * 4
                    bbox_list.append(bbox)
                    scale_list.append(scales)

                structure_list.append(text)
                score_list.append(structure_probs[batch_idx, idx])
            structure_batch_list.append(structure_list)
            structure_score = np.mean(score_list)

        bbox_batch_array = np.multiply(np.array(bbox_list), np.array(scale_list))
        bbox_batch_list = [bbox_batch_array.astype(int).tolist()]

        return bbox_batch_list, structure_batch_list, structure_score

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
                    bbox_list.append(bbox.astype(int))
            structure_batch_list.append(structure_list)
            bbox_batch_list.append(bbox_list)
        return bbox_batch_list, structure_batch_list

    def _get_bbox_scales(self, padding_shape, ori_shape):
        if self.model_name == "SLANet":
            w, h = ori_shape
            return w, h
        else:
            w, h = padding_shape
            ori_w, ori_h = ori_shape
            ratio_w = w / ori_w
            ratio_h = h / ori_h
            ratio = min(ratio_w, ratio_h)
            return w / ratio, h / ratio
