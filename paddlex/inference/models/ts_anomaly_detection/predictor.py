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

import os
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from ....modules.ts_anomaly_detection.model_list import MODELS
from ...common.batch_sampler import TSBatchSampler
from ...common.reader import ReadTS
from ..base import BasePredictor
from ..common import (
    BuildTSDataset,
    TimeFeature,
    TSCutOff,
    TSNormalize,
    TStoArray,
    TStoBatch,
)
from .processors import GetAnomaly
from .result import TSAdResult


class TSAdPredictor(BasePredictor):
    """TSAdPredictor that inherits from BasePredictor."""

    entities = MODELS

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes TSAdPredictor.

        Args:
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.preprocessors, self.infer, self.postprocessors = self._build()

    def _build_batch_sampler(self) -> TSBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return TSBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the result class, TopkResult.

        Returns:
            type: The TopkResult class.
        """
        return TSAdResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        preprocessors = {
            "ReadTS": ReadTS(),
            "TSCutOff": TSCutOff(self.config["size"]),
        }

        if self.config.get("scale", None):
            scaler_file_path = os.path.join(self.model_dir, "scaler.pkl")
            if not os.path.exists(scaler_file_path):
                raise Exception(f"Cannot find scaler file: {scaler_file_path}")
            preprocessors["TSNormalize"] = TSNormalize(
                scaler_file_path, self.config["info_params"]
            )

        preprocessors["BuildTSDataset"] = BuildTSDataset(self.config["info_params"])

        if self.config.get("time_feat", None):
            preprocessors["TimeFeature"] = TimeFeature(
                self.config["info_params"],
                self.config["size"],
                self.config["holiday"],
            )
        preprocessors["TStoArray"] = TStoArray(self.config["input_data"])
        preprocessors["TStoBatch"] = TStoBatch()
        infer = self.create_static_infer()
        postprocessors = {}
        postprocessors["GetAnomaly"] = GetAnomaly(
            self.config["model_threshold"], self.config["info_params"]
        )
        return preprocessors, infer, postprocessors

    def process(self, batch_data: List[Union[str, pd.DataFrame]]) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, pd.DataFrame], ...]): A batch of input data (e.g., image file paths).

        Returns:
            dict: A dictionary containing the input path, raw image, class IDs, scores, and label names for every instance of the batch. Keys include 'input_path', 'input_img', 'class_ids', 'scores', and 'label_names'.
        """

        batch_raw_ts = self.preprocessors["ReadTS"](ts_list=batch_data.instances)
        batch_cutoff_ts = self.preprocessors["TSCutOff"](ts_list=batch_raw_ts)

        if "TSNormalize" in self.preprocessors:
            batch_ts = self.preprocessors["TSNormalize"](ts_list=batch_cutoff_ts)
            batch_input_ts = self.preprocessors["BuildTSDataset"](ts_list=batch_ts)
        else:
            batch_input_ts = self.preprocessors["BuildTSDataset"](
                ts_list=batch_cutoff_ts
            )

        if "TimeFeature" in self.preprocessors:
            batch_ts = self.preprocessors["TimeFeature"](ts_list=batch_input_ts)
            batch_ts = self.preprocessors["TStoArray"](ts_list=batch_ts)
        else:
            batch_ts = self.preprocessors["TStoArray"](ts_list=batch_input_ts)

        x = self.preprocessors["TStoBatch"](ts_list=batch_ts)
        batch_preds = self.infer(x=x)

        batch_ts_preds = self.postprocessors["GetAnomaly"](
            ori_ts_list=batch_input_ts, pred_list=batch_preds
        )
        return {
            "input_path": batch_data.input_paths,
            "input_ts": batch_raw_ts,
            "anomaly": batch_ts_preds,
        }
