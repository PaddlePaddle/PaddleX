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
from .base import BasePipeline
from .single_model_pipeline import SingleModelPipeline
from .ocr import OCRPipeline
from .table_recognition import TableRecPipeline


def create_pipeline(
    pipeline: str,
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
    if not Path(pipeline).exists():
        # XXX: using dict class to handle all pipeline configs
        pipeline = (
            Path(__file__).parent.parent.parent / "pipelines" / f"{pipeline}.yaml"
        )
        if not Path(pipeline).exists():
            raise Exception(f"The pipeline don't exist! ({pipeline})")
    config = parse_config(pipeline)
    pipeline_name = config["Global"]["pipeline_name"]
    predictor_kwargs = {"use_hpip": use_hpip}
    if hpi_params is not None:
        predictor_kwargs["hpi_params"] = hpi_params
    pipeline = BasePipeline.get(pipeline_name)(
        predictor_kwargs=predictor_kwargs, *args, **config["Pipeline"], **kwargs
    )
    return pipeline
