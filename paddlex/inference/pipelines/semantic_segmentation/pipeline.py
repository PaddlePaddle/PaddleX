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
from ...models import HPIConfig, PaddlePredictorOption
from ...models.semantic_segmentation.result import SegResult
from ...utils.benchmark import benchmark
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline


@benchmark.time_methods
class _SemanticSegmentationPipeline(BasePipeline):
    """Semantic Segmentation Pipeline"""

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
        """Initializes the semantic segmentation pipeline.

        Args:
            config (Dict): Configuration dictionary containing model and other parameters.
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
        **kwargs,
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
