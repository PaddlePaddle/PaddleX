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

from ....modules.object_detection.model_list import LAYOUTANALYSIS_MODELS
from ..bindings import create_binding_registration, register_predictor_binding_map
from ..runners import create_pretrained_dynamic_runner_builder
from .predictor import (
    LAYOUT_ANALYSIS_TRANSFORMERS_MODELS,
    LayoutAnalysisRunnerPredictor,
    LayoutAnalysisTransformersPredictor,
)


def _load_ppdoclayoutv2():
    from ..object_detection.modeling import PPDocLayoutV2

    return PPDocLayoutV2


def _load_ppdoclayoutv3():
    from ..object_detection.modeling import PPDocLayoutV3

    return PPDocLayoutV3


register_predictor_binding_map(
    LayoutAnalysisRunnerPredictor,
    {
        "paddle_static": LAYOUTANALYSIS_MODELS,
        "paddle_dynamic": (
            create_binding_registration(
                ("PP-DocLayoutV2",),
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_ppdoclayoutv2,
                    use_safetensors=True,
                    convert_from_hf=True,
                    dtype="float32",
                ),
            ),
            create_binding_registration(
                ("PP-DocLayoutV3",),
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_ppdoclayoutv3,
                    use_safetensors=True,
                    convert_from_hf=True,
                    dtype="float32",
                ),
            ),
        ),
        "hpi": LAYOUTANALYSIS_MODELS,
        "onnxruntime": LAYOUTANALYSIS_MODELS,
    },
)
register_predictor_binding_map(
    LayoutAnalysisTransformersPredictor,
    {"transformers": LAYOUT_ANALYSIS_TRANSFORMERS_MODELS},
)

# Backward compatibility
LayoutAnalysisPredictor = LayoutAnalysisRunnerPredictor
