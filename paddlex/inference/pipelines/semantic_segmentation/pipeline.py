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

from typing import Union, Any, Tuple, List, Dict, Optional, Literal
import numpy as np
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

from ...models.semantic_segmentation.result import SegResult


class SemanticSegmentationPipeline(BasePipeline):
    """Semantic Segmentation Pipeline"""

    entities = "semantic_segmentation"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
            device (str): The device to run the prediction on. Default is None.
            pp_option (PaddlePredictorOption): Options for PaddlePaddle predictor. Default is None.
            use_hpip (bool): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        semantic_segmentation_model_config = config["SubModules"][
            "SemanticSegmentation"
        ]
        self.semantic_segmentation_model = self.create_model(
            semantic_segmentation_model_config
        )
        self.target_size = semantic_segmentation_model_config["target_size"]

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        target_size: Union[Literal[-1], None, int, Tuple[int]] = None,
        **kwargs
    ) -> SegResult:
        """Predicts semantic segmentation results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            target_size (Literal[-1] | None | int | tuple[int]): The Image size model used to do prediction. Default is None.
                If it's set to -1, the original image size will be used.
                If it's set to None, the previous level's setting will be used.
                If it's set to an integer value, the image will be rescaled to the size of (value, value).
                If it's set to a tuple of two integers, the image will be rescaled to the size of (height, width).
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            SegResult: The predicted segmentation results.
        """
        yield from self.semantic_segmentation_model(input, target_size=target_size)
