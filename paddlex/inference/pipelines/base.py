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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from ...utils import logging
from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..models import BasePredictor
from ..utils.hpi import HPIConfig
from ..utils.pp_option import PaddlePredictorOption


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
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the class with specified parameters.

        Args:
            device (str, optional): The device to use for prediction. Defaults to None.
            pp_option (PaddlePredictorOption, optional): The options for PaddlePredictor. Defaults to None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """
        super().__init__()
        self.device = device
        self.pp_option = pp_option
        self.use_hpip = use_hpip
        self.hpi_config = hpi_config

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
        # Should we log if the actual parameter to use is different from the default?
        use_hpip = config.get("use_hpip", self.use_hpip)
        hpi_config = config.get("hpi_config", None)
        if self.hpi_config is not None:
            hpi_config = hpi_config or {}
            hpi_config = {**self.hpi_config, **hpi_config}

        from .. import create_predictor

        logging.info("Creating model: %s", (config["model_name"], model_dir))

        # TODO(gaotingquan): support to specify pp_option by model in pipeline
        if self.pp_option is not None:
            pp_option = self.pp_option.copy()
            pp_option.model_name = config["model_name"]
            pp_option.run_mode = self.pp_option.run_mode
        else:
            pp_option = None

        model = create_predictor(
            model_name=config["model_name"],
            model_dir=model_dir,
            device=self.device,
            batch_size=config.get("batch_size", 1),
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
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

        use_hpip = config.get("use_hpip", self.use_hpip)
        hpi_config = config.get("hpi_config", None)
        if self.hpi_config is not None:
            hpi_config = hpi_config or {}
            hpi_config = {**self.hpi_config, **hpi_config}

        pipeline = create_pipeline(
            config=config,
            device=self.device,
            pp_option=(
                self.pp_option.copy() if self.pp_option is not None else self.pp_option
            ),
            use_hpip=use_hpip,
            hpi_config=hpi_config,
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
