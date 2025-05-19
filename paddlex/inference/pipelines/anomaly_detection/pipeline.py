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

import numpy as np

from ....utils.deps import pipeline_requires_extra
from ...models.anomaly_detection.result import UadResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline


class _AnomalyDetectionPipeline(BasePipeline):
    """Image AnomalyDetectionPipeline Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the image anomaly detection pipeline.

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

        anomaly_detetion_model_config = config["SubModules"]["AnomalyDetection"]
        self.anomaly_detetion_model = self.create_model(anomaly_detetion_model_config)

    def predict(
        self, input: Union[str, List[str], np.ndarray, List[np.ndarray]], **kwargs
    ) -> UadResult:
        """Predicts anomaly detection results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            UadResult: The predicted anomaly results.
        """
        yield from self.anomaly_detetion_model(input)


@pipeline_requires_extra("cv")
class AnomalyDetectionPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "anomaly_detection"

    @property
    def _pipeline_cls(self):
        return _AnomalyDetectionPipeline

    def _get_batch_size(self, config):
        return config["SubModules"]["AnomalyDetection"].get("batch_size", 1)
