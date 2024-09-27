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

from ...modules.ts_anomaly_detection.model_list import MODELS
from ..components import *
from ..results import TSAdResult
from .base import TSPredictor


class TSAdPredictor(TSPredictor):

    entities = MODELS

    def _build_components(self):
        if not self.config.get("info_params", None):
            raise Exception("info_params is not found in config file")

        self._add_component(ReadTS())
        self._add_component(TSCutOff(self.config["size"]))

        if self.config.get("scale", None):
            scaler_file_path = os.path.join(self.model_dir, "scaler.pkl")
            if not os.path.exists(scaler_file_path):
                raise Exception(f"Cannot find scaler file: {scaler_file_path}")
            self._add_component(
                TSNormalize(scaler_file_path, self.config["info_params"])
            )

        self._add_component(BuildTSDataset(self.config["info_params"]))

        if self.config.get("time_feat", None):
            self._add_component(
                TimeFeature(
                    self.config["info_params"],
                    self.config["size"],
                    self.config["holiday"],
                )
            )
        self._add_component(TStoArray(self.config["input_data"]))

        predictor = TSPPPredictor(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )
        self._add_component(("Predictor", predictor))

        self._add_component(
            GetAnomaly(self.config["model_threshold"], self.config["info_params"])
        )

    def _pack_res(self, single):
        return TSAdResult({"ts_path": single["ts_path"], "anomaly": single["anomaly"]})
