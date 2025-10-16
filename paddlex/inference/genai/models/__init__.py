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

import contextlib
import importlib
from pathlib import Path
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from ....utils import logging
from ...utils.official_models import official_models
from ..utils import check_backend, model_name_to_module_name

NETWORK_CLASS_GETTER_KEY = "get_network_class"
PROCESSOR_CLASS_GETTER_KEY = "get_processor_class"
CONFIG_GETTER_KEY = "get_config"
CHAT_TEMPLATE_PATH_GETTER_KEY = "get_chat_template_path"
DEFAULT_CHAT_TEMPLATE_FILENAME = "chat_template.jinja"

ALL_MODEL_NAMES = {"PaddleOCR-VL-0.9B"}


def _check_model_name_and_backend(model_name, backend):
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}")

    check_backend(backend)


def get_model_dir(model_name, backend):
    _check_model_name_and_backend(model_name, backend)

    try:
        model_dir = official_models[model_name]
    except Exception as e:
        raise RuntimeError(
            f"Could not prepare the official model for the {repr(model_name)} model with the {repr(backend)} backend."
        ) from e

    return str(model_dir)


def get_model_components(model_name, backend):
    def _get_component(getter_key):
        if not hasattr(model_module, getter_key):
            raise RuntimeError(f"`{model_module}` does not have `{getter_key}`")
        getter = getattr(model_module, getter_key)
        comp = getter(backend)
        return comp

    _check_model_name_and_backend(model_name, backend)

    mod_name = model_name_to_module_name(model_name)

    try:
        model_module = importlib.import_module(f".{mod_name}", package=__package__)
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown model: {model_name}") from e

    network_class = _get_component(NETWORK_CLASS_GETTER_KEY)

    if backend == "sglang":
        processor_class = _get_component(PROCESSOR_CLASS_GETTER_KEY)
    else:
        processor_class = None

    return network_class, processor_class


def get_default_config(model_name, backend):
    _check_model_name_and_backend(model_name, backend)

    mod_name = model_name_to_module_name(model_name)

    try:
        config_module = importlib.import_module(
            f"..configs.{mod_name}", package=__package__
        )
    except ModuleNotFoundError:
        logging.debug("No default configs were found for the model '%s'", model_name)
        default_config = {}
    else:
        if not hasattr(config_module, CONFIG_GETTER_KEY):
            raise RuntimeError(f"`{config_module}` does not have `{CONFIG_GETTER_KEY}`")
        config_getter = getattr(config_module, CONFIG_GETTER_KEY)
        default_config = config_getter(backend)

    return default_config


@contextlib.contextmanager
def get_chat_template_path(model_name, backend, model_dir):
    _check_model_name_and_backend(model_name, backend)

    with importlib.resources.path(
        "paddlex.inference.genai.chat_templates", f"{model_name}.jinja"
    ) as chat_template_path:
        if not chat_template_path.exists():
            default_chat_template_path = Path(model_dir, DEFAULT_CHAT_TEMPLATE_FILENAME)
            if (
                default_chat_template_path.exists()
                and default_chat_template_path.is_file()
            ):
                # TODO: Support symbolic links
                yield default_chat_template_path
            else:
                logging.debug(
                    "No chat template was found for the model '%s' with the backend '%s'",
                    model_name,
                    backend,
                )
                yield None
        else:
            yield chat_template_path
