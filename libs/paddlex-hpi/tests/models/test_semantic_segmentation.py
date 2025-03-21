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
import pytest
from paddlex_hpi.models import SegPredictor
from tests.models.base import BaseTestPredictor

from paddlex.inference.results import SegResult

MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/seg_model.zip"
INPUT_DATA_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/seg_input.png"
EXPECTED_RESULT_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/seg_result.json"


class TestSegPredictor(BaseTestPredictor):
    @property
    def model_url(self):
        return MODEL_URL

    @property
    def input_data_url(self):
        return INPUT_DATA_URL

    @property
    def expected_result_url(self):
        return EXPECTED_RESULT_URL

    @property
    def expected_result_with_args_url(self):
        return EXPECTED_RESULT_URL

    @property
    def predictor_cls(self):
        return SegPredictor

    @property
    def should_test_with_args(self):
        return True

    def _predict_with_predictor_args(
        self, model_path, input_data_path, device, expected_result_with_args
    ):
        with pytest.raises(TypeError):
            predictor = self.predictor_cls(model_path, device=device, target_size=400)

    def _predict_with_predict_args(
        self,
        model_path,
        input_data_path,
        device,
        expected_result,
        expected_result_with_args,
    ):
        predictor = self.predictor_cls(model_path, device=device)
        with pytest.raises(TypeError):
            output = predictor(str(input_data_path), target_size=400)
            output = list(output)

    def _check_result(self, result, expected_result):
        assert isinstance(result, SegResult)
        assert "input_img" in result
        result.pop("input_img")
        assert set(result) == set(expected_result)
        pred = result["pred"]
        expected_pred = np.array(expected_result["pred"], dtype=np.int32)
        assert pred.shape == expected_pred.shape
        assert (pred != expected_pred).sum() / pred.size < 0.01
