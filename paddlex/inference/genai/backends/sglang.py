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

import json
import os
import subprocess
import sys
import tempfile
import textwrap

from ....utils.deps import require_genai_engine_plugin


def run_sglang_server(host, port, model_name, model_dir, config, chat_template_path):
    require_genai_engine_plugin("sglang-server")

    data = json.dumps(
        {
            "host": host,
            "port": port,
            "model_name": model_name,
            "model_dir": model_dir,
            "config": config,
            "chat_template_path": str(chat_template_path),
        }
    )

    # HACK
    code = textwrap.dedent(
        f"""
    import json
    import os

    from paddlex.inference.genai.configs.utils import (
        backend_config_to_args,
        set_config_defaults,
        update_backend_config,
    )
    from paddlex.inference.genai.models import get_model_components
    from sglang.srt.configs.model_config import multimodal_model_archs
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    data = json.loads({repr(data)})

    host = data["host"]
    port = data["port"]
    model_name = data["model_name"]
    model_dir = data["model_dir"]
    config = data["config"]
    chat_template_path = data["chat_template_path"]

    network_class, processor_class = get_model_components(model_name, "sglang")

    ModelRegistry.models[network_class.__name__] = network_class
    multimodal_model_archs.append(network_class.__name__)
    PROCESSOR_MAPPING[network_class] = processor_class

    set_config_defaults(config, {{"served-model-name": model_name}})

    if chat_template_path:
        set_config_defaults(config, {{"chat-template": chat_template_path}})

    set_config_defaults(config, {{"enable-metrics": True}})

    update_backend_config(
        config,
        {{
            "model-path": model_dir,
            "host": host,
            "port": port,
        }},
    )

    if __name__ == "__main__":
        args = backend_config_to_args(config)

        server_args = prepare_server_args(args)

        try:
            launch_server(server_args)
        finally:
            kill_process_tree(os.getpid(), include_parent=False)
    """
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        subprocess.check_call([sys.executable, script_path])
    finally:
        os.unlink(script_path)
