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
from ..engines.paddle import PaddleDynamicEngineSpec
from ..runners import create_pretrained_dynamic_runner_builder
from .predictor import MODELS, WarpRunnerPredictor


def _load_uvdocnet():
    from .modeling import UVDocNet

    return UVDocNet


register_predictor_binding_map(
    WarpRunnerPredictor,
    {
        "paddle_static": MODELS,
        "paddle_dynamic": create_binding_registration(
            WarpRunnerPredictor.entities,
            **{
                PaddleDynamicEngineSpec.BINDING_EXTRA_RUNNER_BUILDER_KEY: create_pretrained_dynamic_runner_builder(
                    _load_uvdocnet,
                    use_safetensors=True,
                    convert_from_hf=True,
                ),
            },
        ),
        "hpi": MODELS,
        "onnxruntime": MODELS,
    },
)

# Backward compatibility
WarpPredictor = WarpRunnerPredictor
