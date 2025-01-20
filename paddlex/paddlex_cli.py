# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import os
import argparse
import subprocess
import sys

from importlib_resources import files, as_file

from . import create_pipeline
from .inference.pipelines import create_pipeline_from_config, load_pipeline_config
from .repo_manager import setup, get_all_supported_repo_names
from .utils import logging
from .utils.interactive_get_pipeline import interactive_get_pipeline
from .utils.pipeline_arguments import PIPELINE_ARGUMENTS


def _install_serving_deps():
    with as_file(files("paddlex").joinpath("serving_requirements.txt")) as req_file:
        return subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        )


def args_cfg():
    """parse cli arguments"""

    def parse_str(s):
        """convert str type value
        to None type if it is "None",
        to bool type if it means True or False.
        """
        if s in ("None", "none", "NONE"):
            return None
        elif s in ("TRUE", "True", "true", "T", "t"):
            return True
        elif s in ("FALSE", "False", "false", "F", "f"):
            return False
        return s

    parser = argparse.ArgumentParser(
        "Command-line interface for PaddleX. Use the options below to install plugins, run pipeline predictions, or start the serving application."
    )

    install_group = parser.add_argument_group("Install PaddleX Options")
    pipeline_group = parser.add_argument_group("Pipeline Predict Options")
    serving_group = parser.add_argument_group("Serving Options")

    ################# install pdx #################
    install_group.add_argument(
        "--install",
        action="store_true",
        default=False,
        help="Install specified PaddleX plugins.",
    )
    install_group.add_argument(
        "plugins",
        nargs="*",
        default=[],
        help="Names of custom development plugins to install (space-separated).",
    )
    install_group.add_argument(
        "--no_deps",
        action="store_true",
        help="Install custom development plugins without their dependencies.",
    )
    install_group.add_argument(
        "--platform",
        type=str,
        choices=["github.com", "gitee.com"],
        default="github.com",
        help="Platform to use for installation (default: github.com).",
    )
    install_group.add_argument(
        "-y",
        "--yes",
        dest="update_repos",
        action="store_true",
        help="Automatically confirm prompts and update repositories.",
    )
    install_group.add_argument(
        "--use_local_repos",
        action="store_true",
        default=False,
        help="Use local repositories if they exist.",
    )

    ################# pipeline predict #################
    pipeline_group.add_argument(
        "--pipeline", type=str, help="Name of the pipeline to execute for prediction."
    )
    pipeline_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input data or path for the pipeline, supports specific file and directory.",
    )
    pipeline_group.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the prediction results.",
    )
    pipeline_group.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the pipeline on (e.g., 'cpu', 'gpu:0').",
    )
    pipeline_group.add_argument(
        "--use_hpip", action="store_true", help="Enable HPIP acceleration if available."
    )
    pipeline_group.add_argument(
        "--get_pipeline_config",
        type=str,
        default=None,
        help="Retrieve the configuration for a specified pipeline.",
    )

    ################# serving #################
    serving_group.add_argument(
        "--serve",
        action="store_true",
        help="Start the serving application to handle requests.",
    )
    serving_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to serve on (default: 0.0.0.0).",
    )
    serving_group.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number to serve on (default: 8080).",
    )

    # Parse known arguments to get the pipeline name
    args, remaining_args = parser.parse_known_args()
    pipeline_name = args.pipeline
    pipeline_args = []

    if not args.install and pipeline_name is not None:

        if pipeline_name not in PIPELINE_ARGUMENTS:
            support_pipelines = ", ".join(PIPELINE_ARGUMENTS.keys())
            logging.error(
                f"Unsupported pipeline: {pipeline_name}, CLI predict only supports these pipelines: {support_pipelines}\n"
            )
            sys.exit(1)

        pipeline_args = PIPELINE_ARGUMENTS[pipeline_name]
        if pipeline_args is None:
            pipeline_args = []
        pipeline_specific_group = parser.add_argument_group(
            f"{pipeline_name.capitalize()} Pipeline Options"
        )
        for arg in pipeline_args:
            pipeline_specific_group.add_argument(
                arg["name"],
                type=parse_str if arg["type"] is bool else arg["type"],
                help=arg.get("help", f"Argument for {pipeline_name} pipeline."),
            )

    return parser, pipeline_args


def install(args):
    """install paddlex"""
    # Enable debug info
    os.environ["PADDLE_PDX_DEBUG"] = "True"
    # Disable eager initialization
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"

    plugins = args.plugins[:]

    if "serving" in plugins:
        plugins.remove("serving")
        _install_serving_deps()
        return

    if plugins:
        repo_names = plugins
    elif len(plugins) == 0:
        repo_names = get_all_supported_repo_names()
    setup(
        repo_names=repo_names,
        no_deps=args.no_deps,
        platform=args.platform,
        update_repos=args.update_repos,
        use_local_repos=args.use_local_repos,
    )
    return


def pipeline_predict(
    pipeline,
    input,
    device,
    save_path,
    use_hpip,
    **pipeline_args,
):
    """pipeline predict"""
    pipeline = create_pipeline(pipeline, device=device, use_hpip=use_hpip)
    result = pipeline.predict(input, **pipeline_args)
    for res in result:
        res.print()
        if save_path:
            res.save_all(save_path=save_path)


def serve(pipeline, *, device, use_hpip, host, port):
    from .inference.pipelines.serving import create_pipeline_app, run_server

    pipeline_config = load_pipeline_config(pipeline)
    pipeline = create_pipeline_from_config(
        pipeline_config, device=device, use_hpip=use_hpip
    )
    app = create_pipeline_app(pipeline, pipeline_config)
    run_server(app, host=host, port=port, debug=False)


# for CLI
def main():
    """API for commad line"""
    parser, pipeline_args = args_cfg()
    args = parser.parse_args()

    if len(sys.argv) == 1:
        logging.warning("No arguments provided. Displaying help information:")
        parser.print_help()
        return

    if args.install:
        install(args)
    elif args.serve:
        serve(
            args.pipeline,
            device=args.device,
            use_hpip=args.use_hpip,
            host=args.host,
            port=args.port,
        )
    else:
        if args.get_pipeline_config is not None:
            interactive_get_pipeline(args.get_pipeline_config, args.save_path)
        else:
            pipeline_args_dict = {}
            from .utils.flags import USE_NEW_INFERENCE
            if USE_NEW_INFERENCE:
                for arg in pipeline_args:
                    arg_name = arg["name"].lstrip("-")
                    if hasattr(args, arg_name):
                        pipeline_args_dict[arg_name] = getattr(args, arg_name)
                    else:
                        logging.warning(f"Argument {arg_name} is missing in args")
            return pipeline_predict(
                args.pipeline,
                args.input,
                args.device,
                args.save_path,
                use_hpip=args.use_hpip,
                **pipeline_args_dict,
            )
