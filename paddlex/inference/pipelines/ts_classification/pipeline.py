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

from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ....utils.deps import pipeline_requires_extra
from ...models import HPIConfig, PaddlePredictorOption
from ...models.ts_classification.result import TSClsResult
from ...utils.benchmark import benchmark
from ..base import BasePipeline


@benchmark.time_methods
@pipeline_requires_extra("ts")
class TSClsPipeline(BasePipeline):
    """TSClsPipeline Pipeline"""

    entities = "ts_classification"

    def __init__(
        self,
        config: Dict,
        *,
        device: Optional[str] = None,
        engine: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
        **kwargs,
    ) -> None:
        """Initializes the time series classification pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (Optional[str], optional): The device to use for prediction. Defaults to `None`.
            engine (Optional[str], optional): Inference engine. Defaults to `None`.
            engine_config (Optional[Dict[str, Any]], optional): Engine-specific config. Defaults to `None`.
            pp_option (Optional[PaddlePredictorOption], optional): Paddle predictor options.
                Defaults to `None`.
            use_hpip (bool, optional): Whether to use HPIP. Defaults to `False`.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                HPIP configuration. Defaults to `None`.
        """
        super().__init__(
            device=device,
            engine=engine,
            engine_config=engine_config,
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
            **kwargs,
        )

        ts_classification_model_config = config["SubModules"]["TSClassification"]
        self.ts_classification_model = self.create_model(ts_classification_model_config)

    def predict(
        self, input: Union[str, List[str], pd.DataFrame, List[pd.DataFrame]], **kwargs
    ) -> TSClsResult:
        """Predicts time series classification results for the given input.

        Args:
            input (Union[str, list[str], pd.DataFrame, list[pd.DataFrame]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TSFcResult: The predicted time series classification results.
        """
        yield from self.ts_classification_model(input)
