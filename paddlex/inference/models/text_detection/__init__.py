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

from ....modules.text_detection.model_list import MODELS
from ..bindings import create_binding_registration, register_predictor_binding_map
from ..runners import create_pretrained_dynamic_runner_builder
from .predictor import (
    TEXT_DET_TRANSFORMERS_MODELS,
    TextDetRunnerPredictor,
    TextDetTransformersPredictor,
)


def _load_ppocrv5_mobile_det():
    from .modeling import PPOCRV5MobileDet

    return PPOCRV5MobileDet


def _load_ppocrv5_server_det():
    from .modeling import PPOCRV5ServerDet

    return PPOCRV5ServerDet


register_predictor_binding_map(
    TextDetRunnerPredictor,
    {
        "paddle_static": MODELS,
        "paddle_dynamic": (
            create_binding_registration(
                ("PP-OCRv5_mobile_det",),
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_ppocrv5_mobile_det,
                    use_safetensors=True,
                    convert_from_hf=True,
                ),
            ),
            create_binding_registration(
                ("PP-OCRv5_server_det",),
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_ppocrv5_server_det,
                    use_safetensors=True,
                    convert_from_hf=True,
                ),
            ),
        ),
        "hpi": MODELS,
        "onnxruntime": MODELS,
    },
)
register_predictor_binding_map(
    TextDetTransformersPredictor,
    {"transformers": TEXT_DET_TRANSFORMERS_MODELS},
)

# Backward compatibility
TextDetPredictor = TextDetRunnerPredictor
