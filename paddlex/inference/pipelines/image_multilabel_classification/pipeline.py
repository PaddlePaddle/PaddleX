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
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ...utils.pp_option import PaddlePredictorOption
from ...utils.hpi import HPIConfig
from ..base import BasePipeline

from ...models.image_multilabel_classification.result import MLClassResult


class ImageMultiLabelClassificationPipeline(BasePipeline):
    """Image Multi Label Classification Pipeline"""

    entities = "image_multilabel_classification"

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

        self.threshold = config["SubModules"]["ImageMultiLabelClassification"].get(
            "threshold", None
        )
        image_multilabel_classification_model_config = config["SubModules"][
            "ImageMultiLabelClassification"
        ]
        self.image_multilabel_classification_model = self.create_model(
            image_multilabel_classification_model_config
        )
        batch_size = image_multilabel_classification_model_config["batch_size"]

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        threshold: Union[float, dict, list, None] = None,
        **kwargs
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
