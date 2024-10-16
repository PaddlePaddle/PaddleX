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

from pathlib import Path
from typing import Any, Dict, Optional

from ...utils.config import parse_config
from ..utils.get_pipeline_path import get_pipeline_path
from .base import BasePipeline
from .single_model_pipeline import (
    _SingleModelPipeline,
    ImageClassification,
    ObjectDetection,
    InstanceSegmentation,
    SemanticSegmentation,
    TSFc,
    TSAd,
    TSCls,
    MultiLableImageClas,
    SmallObjDet,
    AnomalyDetection,
)
from .ocr import OCRPipeline
from .formula_recognition import FormulaRecognitionPipeline
from .table_recognition import TableRecPipeline
from .seal_recognition import SealOCRPipeline
from .ppchatocrv3 import PPChatOCRPipeline
from .layout_parsing import LayoutParsingPipeline


def load_pipeline_config(pipeline: str) -> Dict[str, Any]:
    if not Path(pipeline).exists():
        pipeline_path = get_pipeline_path(pipeline)
        if pipeline_path is None:
            raise Exception(
                f"The pipeline ({pipeline}) does not exist! Please use a pipeline name or a config file path!"
            )
    else:
        pipeline_path = pipeline
    config = parse_config(pipeline_path)
    return config


def create_pipeline_from_config(
    config: Dict[str, Any],
    device=None,
    pp_option=None,
    use_hpip: bool = False,
    hpi_params: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> BasePipeline:
    pipeline_name = config["Global"]["pipeline_name"]
    pipeline_setting = config["Pipeline"]

    predictor_kwargs = {"use_hpip": use_hpip}
    if "use_hpip" in pipeline_setting:
        predictor_kwargs["use_hpip"] = use_hpip
    if hpi_params is not None:
        predictor_kwargs["hpi_params"] = hpi_params
        pipeline_setting.pop("hpi_params", None)
    elif "hpi_params" in pipeline_setting:
        predictor_kwargs["hpi_params"] = pipeline_setting.pop("hpi_params")

    if pp_option is not None:
        predictor_kwargs["pp_option"] = pp_option
        pipeline_setting.pop("pp_option", None)
    elif "pp_option" in pipeline_setting:
        predictor_kwargs["pp_option"] = pipeline_setting.pop("pp_option")

    device = device if device else pipeline_setting.pop("device", None)

    pipeline_setting.update(kwargs)
    pipeline = BasePipeline.get(pipeline_name)(
        device=device, predictor_kwargs=predictor_kwargs, *args, **pipeline_setting
    )

    return pipeline


def create_pipeline(
    pipeline: str,
    device=None,
    pp_option=None,
    use_hpip: bool = False,
    hpi_params: Optional[Dict[str, Any]] = None,
    *args,
    **kwargs,
) -> BasePipeline:
    """build model evaluater

    Args:
        pipeline (str): the pipeline name, that is name of pipeline class

    Returns:
        BasePipeline: the pipeline, which is subclass of BasePipeline.
    """
    config = load_pipeline_config(pipeline)
    return create_pipeline_from_config(
        config,
        device=device,
        pp_option=pp_option,
        use_hpip=use_hpip,
        hpi_params=hpi_params,
        *args,
        **kwargs,
    )
