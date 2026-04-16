#!/usr/bin/env python3
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Base classes for inference engines."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from ....constants import MODEL_FILE_PREFIX
from ....utils.device import parse_device
from ....utils.subclass_register import AutoRegisterABCMetaClass
from ..bindings import Binding
from ..runners.inference_runner import InferenceRunner
from ..runners.paddle_static.config import PaddlePredictorOption
from ..utils.model_paths import LocalModelFormat, get_model_paths

RunnerBuilder = Callable[..., InferenceRunner]


class InferenceEngine(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base class for inference engines."""

    __is_base = True

    @property
    def engine_config_model(self) -> Optional[Type[BaseModel]]:
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def needs_local_model(self) -> bool:
        return True

    def get_supported_model_formats(
        self,
    ) -> Optional[Tuple[LocalModelFormat, ...]]:
        return None

    def normalize_config(
        self,
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, BaseModel]],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self._engine_config_to_dict(cfg)
        prepared = self.prepare_config_dict(
            raw,
            model_name=model_name,
            device=device,
        )
        validated = self.validate_config_dict(prepared)
        return self.post_normalize_config(validated)

    def prepare_config_dict(
        self,
        raw: Dict[str, Any],
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        del model_name, device
        return raw

    def validate_config_dict(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        config_model = self.engine_config_model
        if config_model is None:
            return raw
        try:
            validated = config_model.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"Invalid {self.name} engine_config: {e}") from e
        return validated.model_dump(**self.get_config_dump_kwargs())

    def get_config_dump_kwargs(self) -> Dict[str, Any]:
        return {"exclude_none": True}

    def post_normalize_config(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        return validated

    def to_predictor_config(self, engine_config: Dict[str, Any]) -> Dict[str, Any]:
        return dict(engine_config)

    def resolve_engine_from_model_dir(self, model_dir: Path) -> str:
        del model_dir
        return self.name

    def ensure_model_files(self, model_dir: Path) -> None:
        model_formats = self.get_supported_model_formats()
        if model_formats is None:
            return
        model_paths = get_model_paths(model_dir, MODEL_FILE_PREFIX)
        if not any(model_format in model_paths for model_format in model_formats):
            raise ValueError(
                f"No valid model files were found for engine {self.name!r}."
            )

    def ensure_environment(self) -> None:
        """Check that required dependencies are installed."""

    @staticmethod
    def _apply_device(raw: Dict[str, Any], device: Optional[str]) -> None:
        """Apply device_type and device_id from a device string into raw config."""
        if device:
            device_type, device_ids = parse_device(device)
            raw["device_type"] = device_type
            raw["device_id"] = device_ids[0] if device_ids is not None else None

    @staticmethod
    def _pp_option_to_engine_config(pp_option: PaddlePredictorOption) -> Dict[str, Any]:
        cfg = {}
        for key, value in pp_option.__dict__.items():
            if value is not None:
                cfg[key] = value
        return cfg

    @classmethod
    def _engine_config_to_dict(
        cls,
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, BaseModel]],
    ) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        if isinstance(cfg, PaddlePredictorOption):
            return cls._pp_option_to_engine_config(cfg)
        if isinstance(cfg, BaseModel):
            return cfg.model_dump(exclude_none=True, by_alias=True)
        raise TypeError(
            f"`engine_config` must be dict, Pydantic model, or PaddlePredictorOption, "
            f"but got {type(cfg).__name__}."
        )


class RunnerEngine(InferenceEngine):
    """Inference engines that can build an InferenceRunner."""

    __is_base = True

    def get_default_runner_builder(self) -> Optional[RunnerBuilder]:
        return None

    def get_runner_builder(
        self, binding: Optional[Binding] = None
    ) -> Optional[RunnerBuilder]:
        if binding is not None and binding.runner_binding is not None:
            runner_builder = binding.runner_binding.runner_builder
            if callable(runner_builder):
                return runner_builder
        return self.get_default_runner_builder()

    @abstractmethod
    def validate_runner(self, runner: InferenceRunner) -> None:
        raise NotImplementedError

    def build_runner(
        self,
        *,
        model_name: str,
        model_dir: Optional[Path],
        model_config: Optional[Dict[str, Any]],
        engine_config: Dict[str, Any],
        binding: Optional[Binding] = None,
    ) -> InferenceRunner:
        runner_builder = self.get_runner_builder(binding)
        if not callable(runner_builder):
            raise RuntimeError(
                f"Model {model_name!r} does not provide {self.name} runner metadata."
            )
        runner = runner_builder(
            model_name=model_name,
            model_dir=model_dir,
            model_config=model_config,
            engine_config=engine_config,
            default_builder=self.get_default_runner_builder(),
        )
        self.validate_runner(runner)
        return runner
