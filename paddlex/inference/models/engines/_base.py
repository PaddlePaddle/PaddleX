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

"""Base classes for engine specifications."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

from pydantic import BaseModel, ValidationError

from ....utils import errors
from ....utils.subclass_register import AutoRegisterABCMetaClass
from ...utils.pp_option import PaddlePredictorOption
from ..predictors import BasePredictor


class EngineSpec(ABC, metaclass=AutoRegisterABCMetaClass):
    """Describes how an inference engine integrates with predictors and models."""

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

    @abstractmethod
    def get_base_predictor_cls(self) -> Type[BasePredictor]:
        raise NotImplementedError

    def get_predictor_cls(self, model_name: str) -> Type[BasePredictor]:
        base_predictor = self.get_base_predictor_cls()
        try:
            return base_predictor.get(model_name)
        except errors.ClassNotFoundException as e:
            raise NotImplementedError(
                f"Model {model_name!r} has no predictor registered for engine "
                f"{self.name!r}."
            ) from e

    def get_supported_engines(self, model_name: str) -> Tuple[str, ...]:
        return tuple(
            engine.lower()
            for engine in self.get_predictor_cls(model_name).get_supported_engines()
        )

    def ensure_predictor_support(self, model_name: str) -> None:
        supported = self.get_supported_engines(model_name)
        if self.name.lower() not in supported:
            predictor_cls = self.get_predictor_cls(model_name)
            raise ValueError(
                f"Model {predictor_cls.__name__!r} does not support engine "
                f"{self.name!r}. Supported engines: {list(supported)!r}."
            )

    def normalize_config(
        self,
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, Any]],
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
        del model_dir

    def ensure_environment(
        self,
        *,
        device: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        del device, engine_config

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
        cfg: Optional[Union[Dict[str, Any], PaddlePredictorOption, Any]],
    ) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        if isinstance(cfg, PaddlePredictorOption):
            return cls._pp_option_to_engine_config(cfg)
        if hasattr(cfg, "model_dump"):
            return cfg.model_dump(exclude_none=True, by_alias=True)
        raise TypeError(
            f"`engine_config` must be dict, Pydantic model, or PaddlePredictorOption, "
            f"but got {type(cfg).__name__}."
        )
