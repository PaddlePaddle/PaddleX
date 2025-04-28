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
from ...models.open_vocabulary_segmentation.results import SAMSegResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

Number = Union[int, float]


@pipeline_requires_extra("multimodal")
class OpenVocabularySegmentationPipeline(BasePipeline):
    """Open Vocabulary Segmentation pipeline"""

    entities = "open_vocabulary_segmentation"

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
        **kwargs
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
