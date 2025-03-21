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

import json
import tempfile
from pathlib import Path
from types import GeneratorType

import pytest
from tests.testing_utils.download import download, download_and_extract
from tests.testing_utils.misc import get_filename

NUM_INPUT_FILES = 10
DEVICES = ["cpu", "gpu:0"]
BATCH_SIZES = [1, 2, 4]


class BaseTestPredictor(object):
    @property
    def model_dir(self):
        raise NotImplementedError

    @property
    def model_url(self):
        raise NotImplementedError

    @property
    def input_data_url(self):
        raise NotImplementedError

    @property
    def expected_result_url(self):
        raise NotImplementedError

    @property
    def expected_result_with_args_url(self):
        raise NotImplementedError

    @property
    def predictor_cls(self):
        raise NotImplementedError

    @property
    def should_test_with_args(self):
        return False

    @pytest.fixture(scope="class")
    def data_dir(self):
        with tempfile.TemporaryDirectory() as td:
            yield Path(td)

    @pytest.fixture(scope="class")
    def model_path(self, data_dir):
        download_and_extract(self.model_url, data_dir, "model")
        yield data_dir / "model"

    @pytest.fixture(scope="class")
    def input_data_path(self, data_dir):
        input_data_path = (data_dir / get_filename(self.input_data_url)).with_stem(
            "test"
        )
        download(self.input_data_url, input_data_path)
        yield input_data_path

    @pytest.fixture(scope="class")
    def expected_result(self, data_dir):
        expected_result_path = data_dir / "expected.json"
        download(self.expected_result_url, expected_result_path)
        with open(expected_result_path, "r", encoding="utf-8") as f:
            expected_result = json.load(f)
        yield expected_result

    @pytest.fixture(scope="class")
    def expected_result_with_args(self, data_dir):
        expected_result_with_args_path = data_dir / "expected_with_args.json"
        download(self.expected_result_with_args_url, expected_result_with_args_path)
        with open(expected_result_with_args_path, "r", encoding="utf-8") as f:
            expected_result = json.load(f)
        yield expected_result

    @pytest.mark.parametrize("device", DEVICES)
    def test___call__single_input_data(
        self, model_path, input_data_path, device, expected_result
    ):
        predictor = self.predictor_cls(model_path, device=device)
        output = predictor(str(input_data_path))
        self._check_output(output, expected_result, 1)
        output = predictor([str(input_data_path), str(input_data_path)])
        self._check_output(output, expected_result, 2)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    def test___call__input_batch_data(
        self, model_path, input_data_path, device, batch_size, expected_result
    ):
        predictor = self.predictor_cls(model_path, device=device)
        predictor.set_predictor(batch_size=batch_size)
        output = predictor([str(input_data_path)] * NUM_INPUT_FILES)
        self._check_output(output, expected_result, NUM_INPUT_FILES)

    @pytest.mark.parametrize("device", DEVICES)
    def test__call__with_predictor_args(
        self, model_path, input_data_path, device, request
    ):
        if self.should_test_with_args:
            self._predict_with_predictor_args(
                model_path,
                input_data_path,
                device,
                request.getfixturevalue("expected_result_with_args"),
            )
        else:
            pytest.skip("Skipping test__call__with_predictor_args for this predictor")

    @pytest.mark.parametrize("device", DEVICES)
    def test__call__with_predict_args(
        self,
        model_path,
        input_data_path,
        device,
        expected_result,
        request,
    ):
        if self.should_test_with_args:
            self._predict_with_predict_args(
                model_path,
                input_data_path,
                device,
                expected_result,
                request.getfixturevalue("expected_result_with_args"),
            )
        else:
            pytest.skip("Skipping test__call__with_predict_args for this predictor")

    def _check_output(self, output, expected_result, expected_num_results):
        assert isinstance(output, GeneratorType)
        # Note that this exhausts the generator
        output = list(output)
        assert len(output) == expected_num_results
        for result in output:
            self._check_result(result, expected_result)

    def _check_result(self, result, expected_result):
        raise NotImplementedError

    def _predict_with_predictor_args(
        self, model_path, input_data_path, device, expected_result_with_args
    ):
        raise NotImplementedError

    def _predict_with_predict_args(
        self,
        model_path,
        input_data_path,
        device,
        expected_result,
        expected_result_with_args,
    ):
        raise NotImplementedError
