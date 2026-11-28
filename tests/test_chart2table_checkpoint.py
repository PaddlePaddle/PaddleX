# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from paddlex.inference.models.doc_vlm.modeling import PPChart2TableInference
from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor


@pytest.mark.parametrize(
    "model_type, weights",
    [
        ("GOT", "model_state.pdparams"),
        ("pp_chart2table", "model_state.pdparams"),
        ("pp_chart2table", "model.safetensors"),
    ],
)
def test_chart2table_checkpoint_format(tmp_path, model_type, weights):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": model_type}), encoding="utf-8"
    )
    (tmp_path / weights).touch()
    predictor = DocVLMLocalPredictor.__new__(DocVLMLocalPredictor)
    predictor.model_name = "PP-Chart2Table"
    predictor._model_dir = tmp_path
    predictor._engine_config = {"device": "cpu"}
    predictor.dtype = "float32"
    processor = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=1))
    with patch.object(predictor, "build_processor", return_value=processor):
        with patch.object(PPChart2TableInference, "from_pretrained") as load:
            if model_type == "GOT":
                with pytest.raises(ValueError, match="PP-Chart2Table_safetensors"):
                    predictor._build()
                load.assert_not_called()
            else:
                predictor._build()
                assert load.call_count == 1
                assert load.call_args.kwargs.get("use_safetensors", False) == (
                    weights == "model.safetensors"
                )
