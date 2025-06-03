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
from ...models.doc_vlm.result import DocVLMResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline


@pipeline_requires_extra("multimodal")
class DocUnderstandingPipeline(BasePipeline):
    """Doc Understanding Pipeline"""

    entities = "doc_understanding"

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
