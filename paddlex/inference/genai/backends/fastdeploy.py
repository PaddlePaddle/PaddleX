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

import sys

from ....utils.deps import require_genai_engine_plugin
from ..configs.utils import (
    backend_config_to_args,
    set_config_defaults,
    update_backend_config,
)


def run_fastdeploy_server(
    host, port, model_name, model_dir, config, chat_template_path
):
    require_genai_engine_plugin("fastdeploy-server")

    if chat_template_path:
        set_config_defaults(config, {"chat-template": str(chat_template_path)})

    update_backend_config(
        config,
        {
            "model": model_dir,
            "host": host,
            "port": port,
        },
    )

    args = backend_config_to_args(config)
    sys.argv[1:] = args

    from fastdeploy.entrypoints.openai.api_server import main as run

    run()
