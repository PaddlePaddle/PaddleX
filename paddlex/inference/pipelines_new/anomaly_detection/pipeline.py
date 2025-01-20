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

from typing import Any, Dict, Optional
import numpy as np

from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

# [TODO] 待更新models_new到models
from ...models_new.anomaly_detection.result import UadResult


class AnomalyDetectionPipeline(BasePipeline):
    """Image AnomalyDetectionPipeline Pipeline"""

    entities = "anomaly_detection"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the image anomaly detection pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        anomaly_detetion_model_config = config["SubModules"]["AnomalyDetection"]
        self.anomaly_detetion_model = self.create_model(anomaly_detetion_model_config)

    def predict(
        self, input: str | list[str] | np.ndarray | list[np.ndarray], **kwargs
    ) -> UadResult:
        """Predicts anomaly detection results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            UadResult: The predicted anomaly results.
        """
        yield from self.anomaly_detetion_model(input)
