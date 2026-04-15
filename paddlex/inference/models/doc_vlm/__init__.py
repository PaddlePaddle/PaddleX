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

from ....modules.doc_vlm.model_list import MODELS
from ..bindings import register_predictor_binding_map
from .constants import PADDLEOCR_VL_MODELS
from .predictor import (
    DocVLMGenAIClientPredictor,
    DocVLMLocalPredictor,
    DocVLMTransformersPredictor,
)

register_predictor_binding_map(
    DocVLMLocalPredictor,
    {"paddle_dynamic": MODELS},
)
register_predictor_binding_map(
    DocVLMTransformersPredictor,
    {"transformers": ("PP-Chart2Table",) + PADDLEOCR_VL_MODELS},
)
register_predictor_binding_map(
    DocVLMGenAIClientPredictor,
    {"genai_client": PADDLEOCR_VL_MODELS},
)

# Backward compatibility
DocVLMPredictor = DocVLMLocalPredictor
