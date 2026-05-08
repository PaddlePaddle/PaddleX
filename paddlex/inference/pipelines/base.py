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

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from ...utils import logging
from ...utils.subclass_register import AutoRegisterABCMetaClass
from ..models import BasePredictor, HPIConfig, PaddlePredictorOption
from ..models.common.genai import uses_server_backend


class BasePipeline(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base class for all pipelines.

    This class serves as a foundation for creating various pipelines.
    It includes common attributes and methods that are shared among all
    pipeline implementations.
    """

    __is_base = True

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        engine: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the class with specified parameters.

        Args:
            device (Optional[str], optional): The device to use for prediction. Defaults to `None`.
            engine (Optional[str], optional): Inference engine. Defaults to `None`.
            engine_config (Optional[Dict[str, Any]], optional): Engine-specific config. Defaults to `None`.
            pp_option (Optional[PaddlePredictorOption], optional): Paddle predictor options.
                Defaults to `None`.
            use_hpip (bool, optional): Whether to use HPIP. Defaults to `False`.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional): HPIP configuration.
                Defaults to `None`.
        """
        super().__init__()
        self.device = device
        self.engine = engine
        self.engine_config = engine_config
        self.pp_option = pp_option
        self.use_hpip = use_hpip
        self.hpi_config = hpi_config

    @abstractmethod
    def predict(self, input, **kwargs):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of `predict`.
        Args:
            input: The input data to predict.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError("The method `predict` has not been implemented yet.")

    @staticmethod
    def _resolve_child_engine(
        config: Dict[str, Any], inherited_engine: Optional[str], *, allow_genai: bool
    ) -> tuple[Optional[str], bool]:
        """Resolve the effective child engine.

        Returns a tuple of `(engine, suppress_inherited_engine_defaults)`.

        Same-level `engine` has the highest priority. If a child omits `engine`
        but specifies another engine selector such as `use_hpip`, or a
        `genai_config` that targets a remote server backend, that selector
        should beat the inherited parent `engine` and fall back to local
        auto-resolution instead of reusing the parent engine defaults.
        """
        if "engine" in config:
            child_engine = config.get("engine", None)
            return child_engine, child_engine is None

        has_local_engine_selector = "use_hpip" in config
        if allow_genai and uses_server_backend(config.get("genai_config", None)):
            has_local_engine_selector = True

        if has_local_engine_selector:
            return None, True

        return inherited_engine, False

    def create_model(self, config: Dict, **kwargs) -> BasePredictor:
        """
        Create a model instance based on the given configuration.
        """
        if "model_config_error" in config:
            raise ValueError(config["model_config_error"])

        model_dir = config.get("model_dir", None)
        model_engine, suppress_inherited_engine_defaults = self._resolve_child_engine(
            config, self.engine, allow_genai=True
        )
        model_engine_config = config.get("engine_config", None)
        if self.engine_config is not None and not suppress_inherited_engine_defaults:
            merged = dict(self.engine_config)
            if model_engine_config:
                merged.update(model_engine_config)
            model_engine_config = merged

        use_hpip = config.get("use_hpip", self.use_hpip)
        hpi_config = config.get("hpi_config", None)
        if self.hpi_config is not None:
            hpi_config = hpi_config or {}
            base = (
                self.hpi_config.model_dump(exclude_none=True)
                if hasattr(self.hpi_config, "model_dump")
                else (self.hpi_config if isinstance(self.hpi_config, dict) else {})
            )
            hpi_config = {**base, **hpi_config}

        from ..models import create_predictor

        logging.info(
            "Creating model: %s", (config["model_name"], model_dir, model_engine)
        )

        pp_option = self.pp_option.copy() if self.pp_option is not None else None

        return create_predictor(
            model_name=config["model_name"],
            model_dir=model_dir,
            device=self.device,
            engine=model_engine,
            engine_config=model_engine_config,
            batch_size=config.get("batch_size", 1),
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
            genai_config=config.get("genai_config", None),
            **kwargs,
        )

    def create_pipeline(self, config: Dict, **kwargs) -> "BasePipeline":
        """
        Creates a pipeline based on the provided configuration.
        """
        if "pipeline_config_error" in config:
            raise ValueError(config["pipeline_config_error"])

        from . import create_pipeline

        pipeline_engine, suppress_inherited_engine_defaults = (
            self._resolve_child_engine(config, self.engine, allow_genai=False)
        )

        use_hpip = config.get("use_hpip", self.use_hpip)
        hpi_config = config.get("hpi_config", None)
        if self.hpi_config is not None:
            hpi_config = hpi_config or {}
            base = (
                self.hpi_config.model_dump(exclude_none=True)
                if hasattr(self.hpi_config, "model_dump")
                else (self.hpi_config if isinstance(self.hpi_config, dict) else {})
            )
            hpi_config = {**base, **hpi_config}

        return create_pipeline(
            config=config,
            device=self.device,
            engine=pipeline_engine,
            engine_config=(
                None if suppress_inherited_engine_defaults else self.engine_config
            ),
            pp_option=(self.pp_option.copy() if self.pp_option is not None else None),
            use_hpip=use_hpip,
            hpi_config=hpi_config,
            **kwargs,
        )

    def close(self):
        pass

    def __call__(self, input, **kwargs):
        """
        Calls the `predict` method with the given input and keyword arguments.

        Args:
            input: The input data to be predicted.
            **kwargs: Additional keyword arguments to be passed to the `predict` method.

        Returns:
            The prediction result from the `predict` method.
        """
        return self.predict(input, **kwargs)
