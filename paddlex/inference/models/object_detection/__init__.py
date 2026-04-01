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

from ....modules.object_detection.model_list import MODELS
from ..bindings import create_binding_registration, register_predictor_binding_map
from ..runners import create_pretrained_dynamic_runner_builder
from .predictor import (  # noqa: F401
    DET_TRANSFORMERS_MODELS,
    RTDETR_L_MODELS,
    DetRunnerPredictor,
    DetTransformersPredictor,
)


def _load_rtdetr():
    from .modeling import RTDETR

    return RTDETR


register_predictor_binding_map(
    DetRunnerPredictor,
    {
        "paddle_static": MODELS,
        "paddle_dynamic": create_binding_registration(
            RTDETR_L_MODELS,
            runner_builder=create_pretrained_dynamic_runner_builder(
                _load_rtdetr,
                use_safetensors=True,
                convert_from_hf=True,
                dtype="float32",
            ),
        ),
        "hpi": MODELS,
        "onnxruntime": MODELS,
    },
)
register_predictor_binding_map(
    DetTransformersPredictor,
    {"transformers": DET_TRANSFORMERS_MODELS},
)

# Backward compatibility
DetPredictor = DetRunnerPredictor
