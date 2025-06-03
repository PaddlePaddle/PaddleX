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
from ...models.image_classification.result import TopkResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline


class _ImageClassificationPipeline(BasePipeline):
    """Image Classification Pipeline"""

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

        image_classification_model_config = config["SubModules"]["ImageClassification"]
        model_kwargs = {}
        if (topk := image_classification_model_config.get("topk", None)) is not None:
            model_kwargs = {"topk": topk}
        self.image_classification_model = self.create_model(
            image_classification_model_config, **model_kwargs
        )
        self.topk = image_classification_model_config.get("topk", 5)

    def predict(
        self, input: Union[str, List[str], np.ndarray, List[np.ndarray]], **kwargs
    ) -> TopkResult:
        """Predicts image classification results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TopkResult: The predicted top k results.
        """

        topk = kwargs.pop("topk", self.topk)
        yield from self.image_classification_model(input, topk=topk)


@pipeline_requires_extra("cv")
class ImageClassificationPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "image_classification"

    @property
    def _pipeline_cls(self):
        return _ImageClassificationPipeline

    def _get_batch_size(self, config):
        return config["SubModules"]["ImageClassification"].get("batch_size", 1)
