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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..predictors import create_predictor
from ...utils.subclass_register import AutoRegisterABCMetaClass


def create_pipeline(
    pipeline_name: str,
    model_list: list,
    model_dir_list: list,
    output: str,
    device: str,
    use_hpip: bool,
    hpi_params: Optional[Dict[str, Any]] = None,
) -> "BasePipeline":
    """build model evaluater

    Args:
        pipeline_name (str): the pipeline name, that is name of pipeline class

    Returns:
        BasePipeline: the pipeline, which is subclass of BasePipeline.
    """
    predictor_kwargs = {"use_hpip": use_hpip}
    if hpi_params is not None:
        predictor_kwargs["hpi_params"] = hpi_params
    pipeline = BasePipeline.get(pipeline_name)(
        output=output, device=device, predictor_kwargs=predictor_kwargs
    )
    pipeline.update_model(model_list, model_dir_list)
    pipeline.load_model()
    return pipeline


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Pipeline"""

    __is_base = True

    def __init__(self, predictor_kwargs: Optional[Dict[str, Any]]) -> None:
        super().__init__()
        if predictor_kwargs is None:
            predictor_kwargs = {}
        self._predictor_kwargs = predictor_kwargs

    # alias the __call__() to predict()
    def __call__(self, *args, **kwargs):
        yield from self.predict(*args, **kwargs)

    def _create_predictor(self, *args, **kwargs):
        return create_predictor(*args, **kwargs, **self._predictor_kwargs)
