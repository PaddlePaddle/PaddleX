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

from ....utils import logging
from ..base import BaseComponent


__all__ = ["Topk", "NormalizeFeatures"]


def _parse_class_id_map(class_ids):
    """parse class id to label map file"""
    if class_ids is None:
        return None
    class_id_map = {id: str(lb) for id, lb in enumerate(class_ids)}
    return class_id_map


class Topk(BaseComponent):
    """Topk Transform"""

    INPUT_KEYS = ["pred"]
    OUTPUT_KEYS = [["class_ids", "scores"], ["class_ids", "scores", "label_names"]]
    DEAULT_INPUTS = {"pred": "pred"}
    DEAULT_OUTPUTS = {
        "class_ids": "class_ids",
        "scores": "scores",
        "label_names": "label_names",
    }

    def __init__(self, topk, class_ids=None):
        super().__init__()
        assert isinstance(topk, (int,))
        self.topk = topk
        self.class_id_map = _parse_class_id_map(class_ids)

    def apply(self, pred):
        """apply"""
        cls_pred = pred[0]
        index = cls_pred.argsort(axis=0)[-self.topk :][::-1].astype("int32")
        clas_id_list = []
        score_list = []
        label_name_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(cls_pred[i].item())
            if self.class_id_map is not None:
                label_name_list.append(self.class_id_map[i.item()])
        result = {
            "class_ids": clas_id_list,
            "scores": np.around(score_list, decimals=5),
        }
        if label_name_list is not None:
            result["label_names"] = label_name_list
        return result


class MultiLabelThreshOutput(BaseComponent):

    INPUT_KEYS = ["pred"]
    OUTPUT_KEYS = [["class_ids", "scores"], ["class_ids", "scores", "label_names"]]
    DEAULT_INPUTS = {"pred": "pred"}
    DEAULT_OUTPUTS = {
        "class_ids": "class_ids",
        "scores": "scores",
        "label_names": "label_names",
    }

    def __init__(self, threshold=0.5, class_ids=None, delimiter=None):
        super().__init__()
        assert isinstance(threshold, (float,))
        self.threshold = threshold
        self.delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = _parse_class_id_map(class_ids)

    def apply(self, pred):
        """apply"""
        y = []
        x = pred[0]
        pred_index = np.where(x >= self.threshold)[0].astype("int32")
        index = pred_index[np.argsort(x[pred_index])][::-1]
        clas_id_list = []
        score_list = []
        label_name_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(x[i].item())
            if self.class_id_map is not None:
                label_name_list.append(self.class_id_map[i.item()])
        result = {
            "class_ids": clas_id_list,
            "scores": np.around(score_list, decimals=5),
        }
        if label_name_list is not None:
            result["label_names"] = label_name_list
        return result


class NormalizeFeatures(BaseComponent):
    """Normalize Features Transform"""

    INPUT_KEYS = ["pred"]
    OUTPUT_KEYS = ["feature"]
    DEAULT_INPUTS = {"pred": "pred"}
    DEAULT_OUTPUTS = {"feature": "feature"}

    def apply(self, pred):
        """apply"""
        feas_norm = np.sqrt(np.sum(np.square(pred[0]), axis=0, keepdims=True))
        feature = np.divide(pred[0], feas_norm)
        return {"feature": feature}
