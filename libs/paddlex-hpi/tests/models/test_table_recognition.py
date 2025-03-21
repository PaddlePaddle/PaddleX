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

from paddlex_hpi.models import TablePredictor
from tests.models.base import BaseTestPredictor
from tests.testing_utils.cv import compare_det_results

from paddlex.inference.results import TableRecResult

MODEL_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/table_model.zip"
INPUT_DATA_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/table_input.jpg"
EXPECTED_RESULT_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/paddlex_hpi/tests/models/table_result.json"


class TestTablePredictor(BaseTestPredictor):
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
    def predictor_cls(self):
        return TablePredictor

    def _check_result(self, result, expected_result):
        def _unflatten_poly(poly):
            return [
                [poly[0], poly[1]],
                [poly[2], poly[3]],
                [poly[4], poly[5]],
                [poly[6], poly[7]],
            ]

        assert isinstance(result, TableRecResult)
        assert "input_img" in result
        result.pop("input_img")
        assert set(result) == set(expected_result)
        compare_det_results(
            [_unflatten_poly(poly) for poly in result["bbox"]],
            [_unflatten_poly(poly) for poly in expected_result["bbox"]],
            labels1=result["structure"],
            labels2=expected_result["structure"],
        )
