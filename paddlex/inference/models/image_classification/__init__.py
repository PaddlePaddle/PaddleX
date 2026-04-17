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

from ..bindings import create_binding_registration, register_predictor_binding_map
from ..runners import create_pretrained_dynamic_runner_builder
from .predictor import (
    CLAS_TRANSFORMERS_MODELS,
    HGNETV2_MODELS,
    MODELS,
    PPLCNET_MODELS,
    ClasRunnerPredictor,
    ClasTransformersPredictor,
)


def _load_pplcnet():
    from .modeling import PPLCNet

    return PPLCNet


def _load_hgnetv2():
    from .modeling import HGNetV2ForImageClassification

    return HGNetV2ForImageClassification


register_predictor_binding_map(
    ClasRunnerPredictor,
    {
        "paddle_static": MODELS,
        "paddle_dynamic": [
            create_binding_registration(
                PPLCNET_MODELS,
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_pplcnet,
                    use_safetensors=True,
                    convert_from_hf=True,
                ),
            ),
            create_binding_registration(
                HGNETV2_MODELS,
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_hgnetv2,
                    use_safetensors=True,
                    convert_from_hf=True,
                ),
            ),
        ],
        "hpi": MODELS,
        "onnxruntime": MODELS,
    },
)
register_predictor_binding_map(
    ClasTransformersPredictor,
    {"transformers": CLAS_TRANSFORMERS_MODELS},
)

# Backward compatibility
ClasPredictor = ClasRunnerPredictor
