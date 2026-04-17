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

"""DocVLM inference model group and constants."""

PP_DOCBEE_MODELS = ("PP-DocBee-2B", "PP-DocBee-7B")
PP_DOCBEE2_MODELS = ("PP-DocBee2-3B",)
PP_CHART2TABLE_MODELS = ("PP-Chart2Table",)
PADDLEOCR_VL_MODELS = ("PaddleOCR-VL-0.9B", "PaddleOCR-VL-1.5-0.9B")

# Model group: maps family name to set of model names.
MODEL_GROUP = {
    "PP-DocBee": set(PP_DOCBEE_MODELS),
    "PP-DocBee2": set(PP_DOCBEE2_MODELS),
    "PP-Chart2Table": set(PP_CHART2TABLE_MODELS),
    "PaddleOCR-VL": set(PADDLEOCR_VL_MODELS),
}

# PaddleOCR-VL specific constants
PADDLEOCR_VL_MAX_NEW_TOKENS = 8192
PADDLEOCR_VL_LOCAL_BATCH_SIZE = 1
PADDLEOCR_VL_GENAI_CLIENT_BATCH_SIZE = 8192
