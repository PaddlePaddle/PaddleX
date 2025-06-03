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

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from ....utils.deps import pipeline_requires_extra
from ...models.semantic_segmentation.result import SegResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline


class _SemanticSegmentationPipeline(BasePipeline):
    """Semantic Segmentation Pipeline"""

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


@pipeline_requires_extra("cv")
class SemanticSegmentationPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "semantic_segmentation"

    @property
    def _pipeline_cls(self):
        return _SemanticSegmentationPipeline

    def _get_batch_size(self, config):
        return config["SubModules"]["SemanticSegmentation"].get("batch_size", 1)
