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

from ....utils import logging
from ....utils.deps import is_genai_engine_plugin_available, require_genai_engine_plugin
from ..configs.utils import (
    backend_config_to_args,
    set_config_defaults,
    update_backend_config,
)
from ..models import ALL_MODEL_NAMES, get_model_components


def register_models():
    from vllm import ModelRegistry

    if is_genai_engine_plugin_available("vllm-server"):
        for model_name in ALL_MODEL_NAMES:
            if model_name not in ModelRegistry.get_supported_archs():
                net_cls, _ = get_model_components(model_name, "vllm")
                ModelRegistry.register_model(net_cls.__name__, net_cls)


def run_vllm_server(host, port, model_name, model_dir, config, chat_template_path):
    require_genai_engine_plugin("vllm-server")

    import uvloop
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        cli_env_setup,
        make_arg_parser,
        run_server,
        validate_parsed_serve_args,
    )

    cli_env_setup()
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    set_config_defaults(config, {"served-model-name": model_name})

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

    import torch

    if torch.version.hip is not None and torch.version.cuda is None:
        # For DCU
        if "api-server-count" in config:
            logging.warning(
                "Key 'api-server-count' will be popped as it is not supported"
            )
            config.pop("api-server-count")

    args = backend_config_to_args(config)
    args = parser.parse_args(args)
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
