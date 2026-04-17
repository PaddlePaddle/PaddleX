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

from typing import Any, Dict, Optional, Union

from ....utils.deps import pipeline_requires_extra
from ...models import HPIConfig, PaddlePredictorOption
from ...models.doc_vlm.result import DocVLMResult
from ...utils.benchmark import benchmark
from ..base import BasePipeline


@benchmark.time_methods
@pipeline_requires_extra("multimodal")
class DocUnderstandingPipeline(BasePipeline):
    """Doc Understanding Pipeline"""

    entities = "doc_understanding"

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
        """Initializes the doc understanding pipeline.

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

        doc_understanding_model_config = config.get("SubModules", {}).get(
            "DocUnderstanding",
            {"model_config_error": "config error for doc_understanding_model!"},
        )
        self.doc_understanding_model = self.create_model(doc_understanding_model_config)

    def predict(self, input: Dict, **kwargs) -> DocVLMResult:
        """Predicts doc understanding results for the given input.

        Args:
            input (dict): The input image and query.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            DocVLMResult: The predicted doc understanding results.
        """
        yield from self.doc_understanding_model(input, **kwargs)

    def close(self):
        self.doc_understanding_model.close()
