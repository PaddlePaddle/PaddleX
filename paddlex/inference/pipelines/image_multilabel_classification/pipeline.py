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
from ...models.image_multilabel_classification.result import MLClassResult
from ...utils.benchmark import benchmark
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline


@benchmark.time_methods
class _ImageMultiLabelClassificationPipeline(BasePipeline):
    """Image Multi Label Classification Pipeline"""

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
        """Initializes the image multilabel classification pipeline.

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

        self.threshold = config["SubModules"]["ImageMultiLabelClassification"].get(
            "threshold", None
        )
        image_multilabel_classification_model_config = config["SubModules"][
            "ImageMultiLabelClassification"
        ]
        self.image_multilabel_classification_model = self.create_model(
            image_multilabel_classification_model_config
        )
        image_multilabel_classification_model_config["batch_size"]

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        threshold: Union[float, dict, list, None] = None,
        **kwargs,
    ) -> MLClassResult:
        """Predicts image classification results for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TopkResult: The predicted top k results.
        """

        yield from self.image_multilabel_classification_model(
            input=input,
            threshold=self.threshold if threshold is None else threshold,
        )


@pipeline_requires_extra("cv")
class ImageMultiLabelClassificationPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "image_multilabel_classification"

    @property
    def _pipeline_cls(self):
        return _ImageMultiLabelClassificationPipeline

    def _get_batch_size(self, config):
        return config["SubModules"]["ImageMultiLabelClassification"]["batch_size"]
