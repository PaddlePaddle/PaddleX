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
from ...base.predictor.transforms import image_common
from ...base import BasePredictor
from .keys import SegKeys as K
from . import transforms as T
from .utils import InnerConfig
from ..model_list import MODELS


class SegPredictor(BasePredictor):
    """ SegPredictor """
    entities = MODELS

    def __init__(self, has_prob_map=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_prob_map = has_prob_map

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
        return [K.SEG_MAP]

    def _run(self, batch_input):
        """ run """
        # XXX:
        os.environ.pop("FLAGS_npu_jit_compile", None)
        images = [data[K.IMAGE] for data in batch_input]
        input_ = np.stack(images, axis=0)
        if input_.ndim == 3:
            input_ = input_[:, np.newaxis]
        input_ = input_.astype(dtype=np.float32, copy=False)
        outputs = self._predictor.predict([input_])
        out_maps = outputs[0]
        # In-place update
        pred = batch_input
        for dict_, out_map in zip(pred, out_maps):
            if self.has_prob_map:
                # `out_map` is prob map
                dict_[K.PROB_MAP] = out_map
                dict_[K.SEG_MAP] = np.argmax(out_map, axis=1)
            else:
                # `out_map` is seg map
                dict_[K.SEG_MAP] = out_map
        return pred

    def _get_pre_transforms_from_config(self):
        """ _get_pre_transforms_from_config """
        # If `K.IMAGE` (the decoded image) is found, return a default list of
        # transformation operators for the input (if possible).
        # If `K.IMAGE` (the decoded image) is not found, `K.IM_PATH` (the image
        # path) must be contained in the input. In this case, we infer
        # transformation operators from the config file.
        # In cases where the input contains both `K.IMAGE` and `K.IM_PATH`,
        # `K.IMAGE` takes precedence over `K.IM_PATH`.
        logging.info(
            f"Transformation operators for data preprocessing will be inferred from config file."
        )
        pre_transforms = self.other_src.pre_transforms
        pre_transforms.insert(0, image_common.ReadImage())
        pre_transforms.append(image_common.ToCHWImage())
        return pre_transforms

    def _get_post_transforms_from_config(self):
        """_get_post_transforms_from_config"""
        post_transforms = []
        if not self.disable_print:
            post_transforms.append(T.PrintResult())
        if not self.disable_save:
            post_transforms.extend(
                [T.GeneratePCMap(), T.SaveSegResults(self.output)])
        return post_transforms
