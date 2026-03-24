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

"""Registry for model_name × engine → predictor + extra_info bindings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple, Type

from ....utils import errors
from ..predictors import BasePredictor


class ModelRegistryLookupError(LookupError):
    """Base error for model binding lookups."""


class UnknownModelError(ModelRegistryLookupError):
    """Raised when a model name has no explicit registry entry."""

    def __init__(self, model_name: str, known_models: Sequence[str]):
        self.model_name = model_name
        self.known_models = tuple(known_models)
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        msg = f"Model {self.model_name!r} is not registered."
        if self.known_models:
            msg += f" Known models: {list(self.known_models)!r}."
        return msg


class UnsupportedEngineError(ModelRegistryLookupError):
    """Raised when a registered model has no binding for the requested engine."""

    def __init__(
        self,
        model_name: str,
        engine: str,
        supported_engines: Sequence[str],
    ):
        self.model_name = model_name
        self.engine = engine
        self.supported_engines = tuple(supported_engines)
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        return (
            f"Model {self.model_name!r} does not support engine {self.engine!r}. "
            f"Supported engines: {list(self.supported_engines)!r}."
        )


@dataclass
class Binding:
    """Binding of (model_name, engine) to a predictor class and optional extra_info."""

    predictor: Type[BasePredictor]
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.extra_info = dict(self.extra_info)


@dataclass(frozen=True)
class BindingRegistration:
    """Registration entry: model names + optional extra_info for an engine."""

    model_names: Tuple[str, ...]
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra_info", dict(self.extra_info))


def create_binding_registration(
    model_names: Sequence[str] | str,
    **extra_info: Any,
) -> BindingRegistration:
    """Create a registration entry with optional extra_info (e.g. runner_builder)."""
    if isinstance(model_names, str):
        model_names = (model_names,)
    return BindingRegistration(model_names=tuple(model_names), extra_info=extra_info)


def _normalize_registrations(
    value: Sequence[str] | BindingRegistration | Sequence[BindingRegistration],
) -> Tuple[BindingRegistration, ...]:
    if isinstance(value, BindingRegistration):
        return (value,)
    if isinstance(value, str):
        return (BindingRegistration(model_names=(value,)),)
    items = tuple(value)
    if items and all(isinstance(x, BindingRegistration) for x in items):
        return items
    return (BindingRegistration(model_names=tuple(items)),)


_registry: Dict[str, Dict[str, Binding]] = {}


def register_predictor_binding_map(
    predictor_cls: Type[BasePredictor],
    bindings: Mapping[
        str,
        Sequence[str] | BindingRegistration | Sequence[BindingRegistration],
    ],
) -> None:
    """Register predictor class for (model_name, engine) pairs."""
    for engine, registration in bindings.items():
        engine = engine.lower()
        for reg in _normalize_registrations(registration):
            for model_name in reg.model_names:
                engine_map = _registry.setdefault(model_name, {})
                binding = Binding(
                    predictor=predictor_cls,
                    extra_info=dict(reg.extra_info),
                )
                existing = engine_map.get(engine)
                if existing is not None and (
                    existing.predictor is not predictor_cls
                    or existing.extra_info != binding.extra_info
                ):
                    raise errors.DuplicateRegistrationError(
                        f"Conflicting registration for model {model_name!r} "
                        f"and engine {engine!r}."
                    )
                engine_map[engine] = binding


def get_supported_engines(model_name: str) -> Tuple[str, ...]:
    """Get supported engines for a model. Raises UnknownModelError if not registered."""
    engine_map = _registry.get(model_name)
    if not engine_map:
        raise UnknownModelError(model_name, tuple(_registry))
    return tuple(engine_map)


def try_get_supported_engines(model_name: str) -> Tuple[str, ...] | None:
    """Get supported engines for a model, or None if not registered."""
    engine_map = _registry.get(model_name)
    if not engine_map:
        return None
    return tuple(engine_map)


def get_binding(model_name: str, engine: str) -> Binding:
    """Get binding for (model_name, engine). Raises on unknown model or unsupported engine."""
    engine_map = _registry.get(model_name)
    if not engine_map:
        raise UnknownModelError(model_name, tuple(_registry))
    binding = engine_map.get(engine.lower())
    if binding is None:
        raise UnsupportedEngineError(model_name, engine, tuple(engine_map))
    return binding


def try_get_binding(model_name: str, engine: str) -> Binding | None:
    """Get binding for (model_name, engine), or None if not found."""
    return _registry.get(model_name, {}).get(engine.lower())


def get_predictor_cls(model_name: str, engine: str) -> Type[BasePredictor]:
    """Get predictor class for (model_name, engine)."""
    return get_binding(model_name, engine).predictor
