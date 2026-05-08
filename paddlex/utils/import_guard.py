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

import importlib
import logging

__all__ = ["import_paddle", "import_paddle_module"]


def import_paddle():
    """Import `paddle` without keeping root StreamHandlers it adds."""
    return import_paddle_module("paddle")


def import_paddle_module(module_name):
    """Import a `paddle` module while preserving existing root handlers."""
    if module_name != "paddle" and not module_name.startswith("paddle."):
        raise ValueError(f"Expected a paddle module name, but got {module_name!r}.")

    root_logger = logging.getLogger()
    existing_handler_ids = {id(handler) for handler in root_logger.handlers}
    try:
        return importlib.import_module(module_name)
    finally:
        for handler in root_logger.handlers[:]:
            if id(handler) in existing_handler_ids:
                continue
            if isinstance(handler, logging.StreamHandler):
                root_logger.removeHandler(handler)
                handler.close()
