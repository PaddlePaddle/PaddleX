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

from ...base import BaseTransform
from .keys import ClsKeys as K
from ....utils import logging

__all__ = ["Topk", "NormalizeFeatures"]


class Topk(BaseTransform):
    """ Topk Transform """

    def __init__(self, topk, class_id_map_file=None, delimiter=None):
        super().__init__()
        assert isinstance(topk, (int, ))
        self.topk = topk
        self.delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = self._parse_class_id_map(class_id_map_file)

    def _parse_class_id_map(self, class_id_map_file):
        """ parse class id to label map file """
        if class_id_map_file is None:
            return None
        if not os.path.exists(class_id_map_file):
            logging.warning(
                "Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
            )
            return None

        class_id_map = {}
        with open(class_id_map_file, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                partition = line.split("\n")[0].partition(self.delimiter)
                class_id_map[int(partition[0])] = str(partition[-1])
        return class_id_map

    def apply(self, data):
        """ apply """
        x = data[K.CLS_PRED]
        class_id_map = data[
            K.
            LABELS] if self.class_id_map is None and K.LABELS in data else self.class_id_map
        y = []
        index = x.argsort(axis=0)[-self.topk:][::-1].astype("int32")
        clas_id_list = []
        score_list = []
        label_name_list = []
        for i in index:
            clas_id_list.append(i.item())
            score_list.append(x[i].item())
            if class_id_map is not None:
                label_name_list.append(class_id_map[i.item()])
        result = {
            "class_ids": clas_id_list,
            "scores": np.around(
                score_list, decimals=5).tolist()
        }
        if label_name_list is not None:
            result["label_names"] = label_name_list
        y.append(result)
        data[K.CLS_RESULT] = y
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IM_PATH, K.CLS_PRED]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.CLS_RESULT]


class NormalizeFeatures(BaseTransform):
    """ Normalize Features Transform """

    def apply(self, data):
        """ apply """
        x = data[K.CLS_PRED]
        feas_norm = np.sqrt(np.sum(np.square(x), axis=0, keepdims=True))
        x = np.divide(x, feas_norm)
        data[K.CLS_RESULT] = x
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.IM_PATH, K.CLS_PRED]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.CLS_RESULT]


class PrintResult(BaseTransform):
    """ Print Result Transform """

    def apply(self, data):
        """ apply """
        logging.info("The prediction result is:")
        logging.info(data[K.CLS_RESULT])
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [K.CLS_RESULT]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return []


class LoadLabels(BaseTransform):
    """load label to data
    """

    def __init__(self, labels=None):
        super().__init__()
        self.labels = labels

    def apply(self, data):
        """ apply """
        if self.labels:
            data[K.LABELS] = self.labels
        return data

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return []

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.LABELS]
