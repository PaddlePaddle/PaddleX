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
import importlib.resources
import subprocess
import sys
import shutil
from pathlib import Path

from . import create_pipeline
from .constants import MODEL_FILE_PREFIX
from .inference.pipelines import load_pipeline_config
from .repo_manager import setup, get_all_supported_repo_names
from .utils.flags import FLAGS_json_format_model
from .utils import logging
from .utils.interactive_get_pipeline import interactive_get_pipeline
from .utils.pipeline_arguments import PIPELINE_ARGUMENTS


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
    paddle2onnx_group = parser.add_argument_group("Paddle2ONNX Options")

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
    install_group.add_argument(
        "--deps_to_replace",
        type=str,
        nargs="+",
        default=None,
        help="Replace dependency version when installing from repositories.",
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
        "--use_hpip",
        action="store_true",
        help="Enable HPIP acceleration by default.",
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
    # Serving also uses `--pipeline`, `--device`, and `--use_hpip`

    ################# paddle2onnx #################
    paddle2onnx_group.add_argument(
        "--paddle2onnx",
        action="store_true",
        help="Convert PaddlePaddle model to ONNX format",
    )
    paddle2onnx_group.add_argument(
        "--paddle_model_dir",
        type=str,
        help="Directory containing the PaddlePaddle model",
    )
    paddle2onnx_group.add_argument(
        "--onnx_model_dir",
        type=str,
        help="Output directory for the ONNX model",
    )
    paddle2onnx_group.add_argument(
        "--opset_version", type=int, help="Version of the ONNX opset to use"
    )

    # Parse known arguments to get the pipeline name
    args, remaining_args = parser.parse_known_args()
    pipeline = args.pipeline
    pipeline_args = []

    if not (args.install or args.serve or args.paddle2onnx) and pipeline is not None:
        if os.path.isfile(pipeline):
            pipeline_name = load_pipeline_config(pipeline)["pipeline_name"]
        else:
            pipeline_name = pipeline

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

    def _install_serving_deps():
        with importlib.resources.path(
            "paddlex", "serving_requirements.txt"
        ) as req_file:
            return subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            )

    def _install_paddle2onnx_deps():
        with importlib.resources.path(
            "paddlex", "paddle2onnx_requirements.txt"
        ) as req_file:
            return subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            )

    def _install_hpi_deps(device_type):
        supported_device_types = ["cpu", "gpu", "npu"]
        if device_type not in supported_device_types:
            logging.error(
                "HPI installation failed!\n"
                "Supported device_type: %s. Your input device_type: %s.\n"
                "Please ensure the device_type is correct.",
                supported_device_types,
                device_type,
            )
            sys.exit(2)

        if device_type == "cpu":
            packages = ["ultra-infer-python"]
        elif device_type == "gpu":
            packages = ["ultra-infer-gpu-python"]
        elif device_type == "npu":
            packages = ["ultra-infer-npu-python"]

        with importlib.resources.path("paddlex", "hpip_links.html") as f:
            return subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--find-links",
                    str(f),
                    *packages,
                ]
            )

    # Enable debug info
    os.environ["PADDLE_PDX_DEBUG"] = "True"
    # Disable eager initialization
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"

    plugins = args.plugins[:]

    if "serving" in plugins:
        plugins.remove("serving")
        if plugins:
            logging.error("`serving` cannot be used together with other plugins.")
            sys.exit(2)
        _install_serving_deps()
        return

    if "paddle2onnx" in plugins:
        plugins.remove("paddle2onnx")
        if plugins:
            logging.error("`paddle2onnx` cannot be used together with other plugins.")
            sys.exit(2)
        _install_paddle2onnx_deps()
        return

    hpi_plugins = list(filter(lambda name: name.startswith("hpi-"), plugins))
    if hpi_plugins:
        for i in hpi_plugins:
            plugins.remove(i)
        if plugins:
            logging.error("`hpi` cannot be used together with other plugins.")
            sys.exit(2)
        if len(hpi_plugins) > 1 or len(hpi_plugins[0].split("-")) != 2:
            logging.error(
                "Invalid HPI plugin installation format detected.\n"
                "Correct format: paddlex --install hpi-<device_type>\n"
                "Example: paddlex --install hpi-gpu"
            )
            sys.exit(2)
        device_type = hpi_plugins[0].split("-")[1]
        _install_hpi_deps(device_type=device_type)
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
        deps_to_replace=args.deps_to_replace,
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
    from .inference.serving.basic_serving import create_pipeline_app, run_server

    pipeline_config = load_pipeline_config(pipeline)
    pipeline = create_pipeline(config=pipeline_config, device=device, use_hpip=use_hpip)
    app = create_pipeline_app(pipeline, pipeline_config)
    run_server(app, host=host, port=port)


# TODO: Move to another module
def paddle_to_onnx(paddle_model_dir, onnx_model_dir, *, opset_version):
    PD_MODEL_FILE_PREFIX = MODEL_FILE_PREFIX
    PD_PARAMS_FILENAME = f"{MODEL_FILE_PREFIX}.pdiparams"
    ONNX_MODEL_FILENAME = f"{MODEL_FILE_PREFIX}.onnx"
    CONFIG_FILENAME = f"{MODEL_FILE_PREFIX}.yml"
    ADDITIONAL_FILENAMES = ["scaler.pkl"]

    def _check_input_dir(input_dir, pd_model_file_ext):
        if input_dir is None:
            sys.exit("Input directory must be specified")
        if not input_dir.exists():
            sys.exit(f"{input_dir} does not exist")
        if not input_dir.is_dir():
            sys.exit(f"{input_dir} is not a directory")
        model_path = (input_dir / PD_MODEL_FILE_PREFIX).with_suffix(pd_model_file_ext)
        if not model_path.exists():
            sys.exit(f"{model_path} does not exist")
        params_path = input_dir / PD_PARAMS_FILENAME
        if not params_path.exists():
            sys.exit(f"{params_path} does not exist")
        config_path = input_dir / CONFIG_FILENAME
        if not config_path.exists():
            sys.exit(f"{config_path} does not exist")

    def _check_paddle2onnx():
        if shutil.which("paddle2onnx") is None:
            sys.exit("Paddle2ONNX is not available. Please install the plugin first.")

    def _run_paddle2onnx(input_dir, pd_model_file_ext, output_dir, opset_version):
        logging.info("Paddle2ONNX conversion starting...")
        # XXX: To circumvent Paddle2ONNX's bug
        if opset_version is None:
            if pd_model_file_ext == ".json":
                opset_version = 19
            else:
                opset_version = 7
            logging.info("Using default ONNX opset version: %d", opset_version)
        cmd = [
            "paddle2onnx",
            "--model_dir",
            str(input_dir),
            "--model_filename",
            str(Path(PD_MODEL_FILE_PREFIX).with_suffix(pd_model_file_ext)),
            "--params_filename",
            PD_PARAMS_FILENAME,
            "--save_file",
            str(output_dir / ONNX_MODEL_FILENAME),
            "--opset_version",
            str(opset_version),
        ]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Paddle2ONNX conversion failed with exit code {e.returncode}")
        logging.info("Paddle2ONNX conversion succeeded")

    def _copy_config_file(input_dir, output_dir):
        src_path = input_dir / CONFIG_FILENAME
        dst_path = output_dir / CONFIG_FILENAME
        shutil.copy(src_path, dst_path)
        logging.info(f"Copied {src_path} to {dst_path}")

    def _copy_additional_files(input_dir, output_dir):
        for filename in ADDITIONAL_FILENAMES:
            src_path = input_dir / filename
            if not src_path.exists():
                continue
            dst_path = output_dir / filename
            shutil.copy(src_path, dst_path)
            logging.info(f"Copied {src_path} to {dst_path}")

    paddle_model_dir = Path(paddle_model_dir)
    if not onnx_model_dir:
        onnx_model_dir = paddle_model_dir
    onnx_model_dir = Path(onnx_model_dir)
    logging.info(f"Input dir: {paddle_model_dir}")
    logging.info(f"Output dir: {onnx_model_dir}")
    pd_model_file_ext = ".json"
    if not FLAGS_json_format_model:
        if not (paddle_model_dir / f"{PD_MODEL_FILE_PREFIX}.json").exists():
            pd_model_file_ext = ".pdmodel"
    _check_input_dir(paddle_model_dir, pd_model_file_ext)
    _check_paddle2onnx()
    _run_paddle2onnx(paddle_model_dir, pd_model_file_ext, onnx_model_dir, opset_version)
    if not (onnx_model_dir.exists() and onnx_model_dir.samefile(paddle_model_dir)):
        _copy_config_file(paddle_model_dir, onnx_model_dir)
        _copy_additional_files(paddle_model_dir, onnx_model_dir)
    logging.info("Done")


# for CLI
def main():
    """API for command line"""
    parser, pipeline_args = args_cfg()
    args = parser.parse_args()

    if len(sys.argv) == 1:
        logging.warning("No arguments provided. Displaying help information:")
        parser.print_help()
        sys.exit(2)

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
    elif args.paddle2onnx:
        paddle_to_onnx(
            args.paddle_model_dir,
            args.onnx_model_dir,
            opset_version=args.opset_version,
        )
    else:
        if args.get_pipeline_config is not None:
            interactive_get_pipeline(args.get_pipeline_config, args.save_path)
        else:
            pipeline_args_dict = {}

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
