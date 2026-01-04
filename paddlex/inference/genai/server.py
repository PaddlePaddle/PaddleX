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

import argparse
import sys

from ...utils import logging
from ...utils.deps import is_genai_engine_plugin_available
from .configs.utils import load_backend_config, update_backend_config
from .constants import DEFAULT_BACKEND, SUPPORTED_BACKENDS
from .models import get_chat_template_path, get_default_config, get_model_dir


def get_arg_parser():
    parser = argparse.ArgumentParser("PaddleX generative AI server.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--backend", type=str, choices=SUPPORTED_BACKENDS, default=DEFAULT_BACKEND
    )
    parser.add_argument(
        "--backend_config", type=str, help="Path to the backend configuration file."
    )
    return parser


def run_genai_server(args):
    plugin_name = f"{args.backend}-server"
    if not is_genai_engine_plugin_available(plugin_name):
        logging.error(
            f"The '{plugin_name}' plugin is not available. Please install it first."
        )
        sys.exit(1)

    if args.backend == "fastdeploy":
        from .backends.fastdeploy import run_fastdeploy_server

        run_server_func = run_fastdeploy_server
    elif args.backend == "vllm":
        from .backends.vllm import run_vllm_server

        run_server_func = run_vllm_server
    elif args.backend == "sglang":
        from .backends.sglang import run_sglang_server

        run_server_func = run_sglang_server
    else:
        raise AssertionError

    if args.model_dir:
        model_dir = args.model_dir
    else:
        try:
            model_dir = get_model_dir(args.model_name, args.backend)
        except Exception:
            logging.error("Failed to get model directory", exc_info=True)
            sys.exit(1)

    if args.backend_config:
        try:
            backend_config = load_backend_config(args.backend_config)
        except Exception:
            logging.error(
                f"Failed to load backend configuration from file: {args.backend_config}",
                exc_info=True,
            )
            sys.exit(1)
    else:
        backend_config = {}

    try:
        default_config = get_default_config(args.model_name, args.backend)
    except Exception:
        logging.error(
            f"Failed to get default configuration for the model", exc_info=True
        )
        sys.exit(1)
    update_backend_config(
        default_config,
        backend_config,
    )
    backend_config = default_config

    with get_chat_template_path(
        args.model_name, args.backend, model_dir
    ) as chat_template_path:
        run_server_func(
            args.host,
            args.port,
            args.model_name,
            model_dir,
            backend_config,
            chat_template_path,
        )


def main(args=None):
    parser = get_arg_parser()
    args = parser.parse_args(args=args)
    run_genai_server(args)


if __name__ == "__main__":
    main()
