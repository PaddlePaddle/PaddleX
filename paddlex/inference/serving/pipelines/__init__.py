# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from typing import Any, Dict, Final

from fastapi import FastAPI

from .image_classification import create_pipeline_app as create_image_classification_app
from .instance_segmentation import (
    create_pipeline_app as create_instance_segmentation_app,
)
from .object_detection import create_pipeline_app as create_object_detection_app
from .ocr import create_pipeline_app as create_ocr_app
from .semantic_segmentation import (
    create_pipeline_app as create_semantic_segmentation_app,
)
from .table_recognition import create_pipeline_app as create_table_recognition_app
from ..app import AppConfig


SERVING_CONFIG_KEY: Final[str] = "Serving"


def create_pipeline_app(pipeline_name: str, pipeline_config: Dict[str, Any]) -> FastAPI:
    if SERVING_CONFIG_KEY not in pipeline_config:
        raise ValueError("Serving config not found")
    app_config = AppConfig.model_validate(pipeline_config[SERVING_CONFIG_KEY])
    if pipeline_name == "image_classification":
        return create_image_classification_app(app_config)
    elif pipeline_name == "instance_segmentation":
        return create_instance_segmentation_app(app_config)
    elif pipeline_name == "object_detection":
        return create_object_detection_app(app_config)
    elif pipeline_name == "ocr":
        return create_ocr_app(app_config)
    elif pipeline_name == "semantic_segmentation":
        return create_semantic_segmentation_app(app_config)
    elif pipeline_name == "table_recognition":
        return create_table_recognition_app(app_config)
    else:
        raise ValueError("Unknown pipeline name")
