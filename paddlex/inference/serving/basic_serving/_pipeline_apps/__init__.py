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

import importlib
from typing import Any, Dict

from .....utils.deps import function_requires_deps, is_dep_available
from ...infra.config import create_app_config

if is_dep_available("fastapi"):
    from fastapi import FastAPI


def _pipeline_name_to_mod_name(pipeline_name: str) -> str:
    if not pipeline_name:
        raise ValueError("Empty pipeline name")
    mod_name = pipeline_name.lower().replace("-", "_")
    if mod_name[0].isdigit():
        return "m_" + mod_name
    return mod_name


# XXX: A dynamic approach is used here for writing fewer lines of code, at the
# cost of sacrificing some benefits of type hints.
@function_requires_deps("fastapi")
def create_pipeline_app(pipeline: Any, pipeline_config: Dict[str, Any]) -> "FastAPI":
    pipeline_name = pipeline_config["pipeline_name"]
    mod_name = _pipeline_name_to_mod_name(pipeline_name)
    mod = importlib.import_module(f".{mod_name}", package=__package__)
    app_config = create_app_config(pipeline_config)
    app_creator = getattr(mod, "create_pipeline_app")
    app = app_creator(pipeline, app_config)
    return app
