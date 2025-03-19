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

from typing import Any, Dict, Optional, Union, List
import numpy as np
from ...utils.pp_option import PaddlePredictorOption
from ...utils.hpi import HPIConfig
from ..base import BasePipeline

from ...models.object_detection.result import DetResult


class RotatedObjectDetectionPipeline(BasePipeline):
    """Rotated Object Detection Pipeline"""

    entities = "rotated_object_detection"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """
        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        rotated_object_detection_model_config = config["SubModules"][
            "RotatedObjectDetection"
        ]
        self.rotated_object_detection_model = self.create_model(
            rotated_object_detection_model_config
        )
        self.threshold = rotated_object_detection_model_config["threshold"]

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        threshold: Union[None, Dict[int, float], float] = None,
        **kwargs
    ) -> DetResult:
        """Predicts rotated object detection results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            threshold (Optional[float]): The threshold value to filter out low-confidence predictions. Default is None.
                If None, it will use the default threshold specified during initialization.
                If a dictionary is provided, it should have integer keys corresponding to the class IDs and float values
                representing the respective thresholds for each class.
                If a single float value is provided, it will be used as the threshold for all classes.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            DetResult: The predicted rotated object detection results.
        """
        yield from self.rotated_object_detection_model(input, threshold=threshold)
