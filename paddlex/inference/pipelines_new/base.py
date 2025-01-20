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

from pathlib import Path
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod
import yaml
import codecs
from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..utils.pp_option import PaddlePredictorOption
from ..models import BasePredictor


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base class for all pipelines.

    This class serves as a foundation for creating various pipelines.
    It includes common attributes and methods that are shared among all
    pipeline implementations.
    """

    __is_base = True

    def __init__(
        self,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the class with specified parameters.

        Args:
            device (str, optional): The device to use for prediction. Defaults to None.
            pp_option (PaddlePredictorOption, optional): The options for PaddlePredictor. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__()
        self.device = device
        self.pp_option = pp_option
        self.use_hpip = use_hpip

    @abstractmethod
    def predict(self, input, **kwargs):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of predict.
        Args:
            input: The input data to predict.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError("The method `predict` has not been implemented yet.")

    def create_model(self, config: Dict, **kwargs) -> BasePredictor:
        """
        Create a model instance based on the given configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.
            **kwargs: The model arguments that needed to be pass.

        Returns:
            BasePredictor: An instance of the model.
        """
        if "model_config_error" in config:
            raise ValueError(config["model_config_error"])

        model_dir = config.get("model_dir", None)
        hpi_params = config.get("hpi_params", None)

        from .. import create_predictor

        model = create_predictor(
            model_name=config["model_name"],
            model_dir=model_dir,
            device=self.device,
            pp_option=self.pp_option,
            use_hpip=self.use_hpip,
            **kwargs,
        )
        return model

    def create_pipeline(self, config: Dict):
        """
        Creates a pipeline based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the pipeline configuration.

        Returns:
            BasePipeline: An instance of the created pipeline.
        """
        if "pipeline_config_error" in config:
            raise ValueError(config["pipeline_config_error"])

        from . import create_pipeline

        pipeline_name = config["pipeline_name"]
        pipeline = create_pipeline(
            pipeline_name,
            config=config,
            device=self.device,
            pp_option=self.pp_option,
            use_hpip=self.use_hpip,
        )
        return pipeline

    def __call__(self, input, **kwargs):
        """
        Calls the predict method with the given input and keyword arguments.

        Args:
            input: The input data to be predicted.
            **kwargs: Additional keyword arguments to be passed to the predict method.

        Returns:
            The prediction result from the predict method.
        """
        return self.predict(input, **kwargs)
