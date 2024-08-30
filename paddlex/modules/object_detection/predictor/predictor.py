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

from ....utils import logging
from ...base import BasePredictor
from ...base.predictor.transforms import image_common
from . import transforms as T
from .keys import DetKeys as K
from .utils import InnerConfig
from ..model_list import MODELS


class DetPredictor(BasePredictor):
    """ Detection Predictor """
    entities = MODELS

    def load_other_src(self):
        """ load the inner config file """
        infer_cfg_file_path = os.path.join(self.model_dir, 'inference.yml')
        if not os.path.exists(infer_cfg_file_path):
            raise FileNotFoundError(
                f"Cannot find config file: {infer_cfg_file_path}")
        return InnerConfig(infer_cfg_file_path)

    @classmethod
    def get_input_keys(cls):
        """ get input keys """
        return [[K.IMAGE], [K.IM_PATH]]

    @classmethod
    def get_output_keys(cls):
        """ get output keys """
        return [K.BOXES]

    def _run(self, batch_input):
        """ run """
        input_dict = {}
        input_dict["image"] = np.stack(
            [data[K.IMAGE] for data in batch_input], axis=0).astype(
                dtype=np.float32, copy=False)
        input_dict["scale_factor"] = np.stack(
            [data[K.SCALE_FACTOR][::-1] for data in batch_input],
            axis=0).astype(
                dtype=np.float32, copy=False)
        input_dict["im_shape"] = np.stack(
            [data[K.IM_SIZE][::-1] for data in batch_input], axis=0).astype(
                dtype=np.float32, copy=False)

        input_ = [input_dict[i] for i in self._predictor.get_input_names()]

        batch_np_boxes, batch_np_boxes_num = self._predictor.predict(input_)

        pred = batch_input
        box_idx_start = 0
        for idx in range(len(batch_input)):
            np_boxes_num = batch_np_boxes_num[idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = batch_np_boxes[box_idx_start:box_idx_end]
            box_idx_start = box_idx_end

            batch_input[idx][K.BOXES] = np_boxes
        return pred

    def _get_pre_transforms_from_config(self):
        """ get preprocess transforms """
        logging.info(
            f"Transformation operators for data preprocessing will be inferred from config file."
        )
        pre_transforms = self.other_src.pre_transforms
        pre_transforms.insert(0, image_common.ReadImage(format='RGB'))
        return pre_transforms

    def _get_post_transforms_from_config(self):
        """get postprocess transforms"""
        post_transforms = []
        if not self.disable_print:
            post_transforms.append(T.PrintResult())
        if not self.disable_save:
            post_transforms.append(
                T.SaveDetResults(
                    save_dir=self.output, labels=self.other_src.labels))
        return post_transforms
