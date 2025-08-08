# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ....utils import logging
from ...utils.official_models import official_models
from ..utils import check_backend

NETWORK_CLASS_GETTER_KEY = "get_network_class"
CONFIG_GETTER_KEY = "get_config"


class GenAIModel(BaseModel):
    name: str
    path: Path
    network_class: Type
    default_config: Optional[Dict[str, Any]] = None


def build_model(model_name, backend, model_dir=None):
    if "." in model_name:
        raise ValueError(f"Illegal model name: {model_name}")

    check_backend(backend)

    if model_dir is None:
        if backend in ("vllm", "sglang"):
            suffix = "_paddle"
        else:
            suffix = "_torch"
        try:
            model_dir = official_models[model_name + suffix]
        except Exception as e:
            raise RuntimeError(
                f"Could not prepare the official model for the {repr(model_name)} model with the {repr(backend)} backend."
            )

    try:
        model_module = importlib.import_module(f".{model_name}", package=__package__)
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown model: {model_name}") from e
    if not hasattr(model_module, NETWORK_CLASS_GETTER_KEY):
        raise RuntimeError(
            f"`{model_module}` does not have `{NETWORK_CLASS_GETTER_KEY}`"
        )
    network_class_getter = getattr(model_module, NETWORK_CLASS_GETTER_KEY)
    network_class = network_class_getter(backend)

    try:
        config_module = importlib.import_module(
            f"..configs.{model_name}", package=__package__
        )
    except ModuleNotFoundError:
        logging.debug("No default configs were found for the model '%s'", model_name)
        default_config = None
    else:
        if not hasattr(config_module, CONFIG_GETTER_KEY):
            raise RuntimeError(f"`{config_module}` does not have `{CONFIG_GETTER_KEY}`")
        config_getter = getattr(config_module, CONFIG_GETTER_KEY)
        default_config = config_getter(backend, model_dir)

    return GenAIModel(
        name=model_name,
        path=model_dir,
        network_class=network_class,
        default_config=default_config,
    )
