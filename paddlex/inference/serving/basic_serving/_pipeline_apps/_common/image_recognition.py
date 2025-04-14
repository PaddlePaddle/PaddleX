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

import uuid
from typing import Final

from ....infra.storage import create_storage
from ..._app import AppContext

DEFAULT_INDEX_DIR: Final[str] = ".index"


def update_app_context(app_context: AppContext) -> None:
    if app_context.config.extra and "index_storage" in app_context.config.extra:
        app_context.extra["index_storage"] = create_storage(
            app_context.config.extra["index_storage"]
        )
    else:
        app_context.extra["index_storage"] = create_storage(
            {"type": "file_system", "directory": DEFAULT_INDEX_DIR}
        )


def generate_index_key() -> str:
    return str(uuid.uuid4())
