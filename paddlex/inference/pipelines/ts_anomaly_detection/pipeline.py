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
from ...models.ts_anomaly_detection.result import TSAdResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline


@pipeline_requires_extra("ts")
class TSAnomalyDetPipeline(BasePipeline):
    """TSAnomalyDetPipeline Pipeline"""

    entities = "ts_anomaly_detection"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the Time Series ad pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """

        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        ts_ad_model_config = config["SubModules"]["TSAnomalyDetection"]
        self.ts_ad_model = self.create_model(ts_ad_model_config)

    def predict(
        self, input: Union[str, List[str], pd.DataFrame, List[pd.DataFrame]], **kwargs
    ) -> TSAdResult:
        """Predicts time series anomaly detection results for the given input.

        Args:
            input (Union[str, list[str], pd.DataFrame, list[pd.DataFrame]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TSAdResult: The predicted time series anomaly detection results.
        """
        yield from self.ts_ad_model(input)
