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

from ....modules.text_recognition.model_list import MODELS
from ..bindings import create_binding_registration, register_predictor_binding_map
from ..runners import create_pretrained_dynamic_runner_builder
from .predictor import (
    TEXT_REC_TRANSFORMERS_MODELS,
    TextRecRunnerPredictor,
    TextRecTransformersPredictor,
)


def _load_ppocrv5_mobile_rec():
    from .modeling import PPOCRV5MobileRec

    return PPOCRV5MobileRec


def _load_ppocrv5_server_rec():
    from .modeling import PPOCRV5ServerRec

    return PPOCRV5ServerRec


register_predictor_binding_map(
    TextRecRunnerPredictor,
    {
        "paddle_static": MODELS,
        "paddle_dynamic": (
            create_binding_registration(
                (
                    "PP-OCRv5_mobile_rec",
                    "eslav_PP-OCRv5_mobile_rec",
                    "korean_PP-OCRv5_mobile_rec",
                    "latin_PP-OCRv5_mobile_rec",
                    "en_PP-OCRv5_mobile_rec",
                    "th_PP-OCRv5_mobile_rec",
                    "el_PP-OCRv5_mobile_rec",
                    "arabic_PP-OCRv5_mobile_rec",
                    "te_PP-OCRv5_mobile_rec",
                    "ta_PP-OCRv5_mobile_rec",
                    "devanagari_PP-OCRv5_mobile_rec",
                    "cyrillic_PP-OCRv5_mobile_rec",
                ),
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_ppocrv5_mobile_rec,
                    use_safetensors=True,
                    convert_from_hf=True,
                    dtype="float32",
                ),
            ),
            create_binding_registration(
                ("PP-OCRv5_server_rec",),
                runner_builder=create_pretrained_dynamic_runner_builder(
                    _load_ppocrv5_server_rec,
                    use_safetensors=True,
                    convert_from_hf=True,
                    dtype="float32",
                ),
            ),
        ),
        "hpi": MODELS,
        "onnxruntime": MODELS,
    },
)
register_predictor_binding_map(
    TextRecTransformersPredictor,
    {"transformers": TEXT_REC_TRANSFORMERS_MODELS},
)

# Backward compatibility
TextRecPredictor = TextRecRunnerPredictor
