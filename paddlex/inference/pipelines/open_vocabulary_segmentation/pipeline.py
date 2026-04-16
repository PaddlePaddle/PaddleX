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
from ...models import HPIConfig, PaddlePredictorOption
from ...models.open_vocabulary_segmentation.results import SAMSegResult
from ...utils.benchmark import benchmark
from ..base import BasePipeline

Number = Union[int, float]


@benchmark.time_methods
@pipeline_requires_extra("multimodal")
class OpenVocabularySegmentationPipeline(BasePipeline):
    """Open Vocabulary Segmentation pipeline"""

    entities = "open_vocabulary_segmentation"

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
        """Initializes the open vocabulary segmentation pipeline.

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

        # create box-prompted SAM-H
        box_prompted_model_cfg = config.get("SubModules", {}).get(
            "BoxPromptSegmentation",
            {"model_config_error": "config error for doc_ori_classify_model!"},
        )
        self.box_prompted_model = self.create_model(box_prompted_model_cfg)

        # create point-prompted SAM-H
        point_prompted_model_cfg = config.get("SubModules", {}).get(
            "PointPromptSegmentation",
            {"model_config_error": "config error for doc_ori_classify_model!"},
        )
        self.point_prompted_model = self.create_model(point_prompted_model_cfg)

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        prompt: Union[List[List[float]], np.ndarray],
        prompt_type: str = "box",
        **kwargs,
    ) -> SAMSegResult:
        """Predicts image segmentation results for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images.
            prompt (list[list[float]] | np.ndarray): The prompt for the input image(s).
            prompt_type (str): The type of prompt, either 'box' or 'point'. Default is 'box'.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            SAMSegResult: The predicted SAM segmentation results.
        """
        if prompt_type == "box":
            yield from self.box_prompted_model(input, prompts={"box_prompt": prompt})
        elif prompt_type == "point":
            yield from self.point_prompted_model(
                input, prompts={"point_prompt": prompt}
            )
        else:
            raise ValueError(
                "Invalid prompt type. Only 'box' and 'point' are supported"
            )
