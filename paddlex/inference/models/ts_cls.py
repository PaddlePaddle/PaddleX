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
from ...modules.ts_classification.model_list import MODELS
from ..components import *
from ..results import TSClsResult
from ..utils.process_hook import batchable_method
from .base import BasicPredictor


class TSClsPredictor(BasicPredictor):

    entities = MODELS

    def _check_args(self, kwargs):
        pass

    def _build_components(self):
        preprocess = self._build_preprocess()
        predictor = TSPPPredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        return {**preprocess, "predictor": predictor, "GetCls": GetCls()}

    def _build_preprocess(self):
        if not self.config.get("info_params", None):
            raise Exception("info_params is not found in config file")

        ops = {}
        ops["ReadTS"] = ReadTS()
        if self.config.get("scale", None):
            scaler_file_path = os.path.join(self.model_dir, "scaler.pkl")
            if not os.path.exists(scaler_file_path):
                raise Exception(f"Cannot find scaler file: {scaler_file_path}")
            ops["TSNormalize"] = TSNormalize(
                scaler_file_path, self.config["info_params"]
            )

        ops["BuildTSDataset"] = BuildTSDataset(self.config["info_params"])
        ops["BuildPadMask"] = BuildPadMask(self.config["input_data"])
        ops["TStoArray"] = TStoArray(self.config["input_data"])

        return ops

    def _pack_res(self, single):
        return TSClsResult(
            {"ts_path": single["ts_path"], "classification": single["classification"]}
        )
