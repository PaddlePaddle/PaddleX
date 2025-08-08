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

from ....utils.deps import require_genai_engine_plugin
from ..configs.utils import backend_config_to_args, update_backend_config


def run_vllm_server(host, port, config, model):
    require_genai_engine_plugin("vllm")

    import uvloop
    from vllm import ModelRegistry
    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        cli_env_setup,
        make_arg_parser,
        run_server,
        validate_parsed_serve_args,
    )

    ModelRegistry.register_model(model.name, model.net)

    cli_env_setup()
    parser = FlexibleArgumentParser()
    parser = make_arg_parser(parser)

    update_backend_config(
        config, model=str(model.path), host=host, port=port, **model.default_config
    )
    args = backend_config_to_args(config)
    args = parser.parse_args(args)
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
