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

import os
import os.path as osp

from ...base.register import register_model_info, register_suite_info
from .config import TextRecConfig
from .model import TextRecModel
from .runner import TextRecRunner

REPO_ROOT_PATH = os.environ.get("PADDLE_PDX_PADDLEOCR_PATH")
PDX_CONFIG_DIR = osp.abspath(osp.join(osp.dirname(__file__), "..", "configs"))

register_suite_info(
    {
        "suite_name": "TextRec",
        "model": TextRecModel,
        "runner": TextRecRunner,
        "config": TextRecConfig,
        "runner_root_path": REPO_ROOT_PATH,
    }
)

register_model_info(
    {
        "model_name": "PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "en_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "en_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)


register_model_info(
    {
        "model_name": "korean_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "korean_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "japan_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "japan_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "chinese_cht_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "chinese_cht_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "te_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "te_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ka_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "ka_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ta_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "ta_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "latin_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "latin_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "arabic_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "arabic_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "cyrillic_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "cyrillic_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "devanagari_PP-OCRv3_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "devanagari_PP-OCRv3_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)


register_model_info(
    {
        "model_name": "PP-OCRv4_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-OCRv4_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-OCRv4_server_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-OCRv4_server_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "en_PP-OCRv4_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "en_PP-OCRv4_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-OCRv4_server_rec_doc",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-OCRv4_server_rec_doc.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "ch_SVTRv2_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "ch_SVTRv2_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
    }
)

register_model_info(
    {
        "model_name": "ch_RepSVTR_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "ch_RepSVTR_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export", "infer"],
    }
)

register_model_info(
    {
        "model_name": "PP-OCRv5_server_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-OCRv5_server_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)

register_model_info(
    {
        "model_name": "PP-OCRv5_mobile_rec",
        "suite": "TextRec",
        "config_path": osp.join(PDX_CONFIG_DIR, "PP-OCRv5_mobile_rec.yaml"),
        "supported_apis": ["train", "evaluate", "predict", "export"],
    }
)
