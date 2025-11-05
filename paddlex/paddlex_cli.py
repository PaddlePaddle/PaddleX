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

import argparse
import ast
import importlib.resources
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import create_pipeline
from .constants import MODEL_FILE_PREFIX
from .inference.pipelines import load_pipeline_config
from .inference.utils.model_paths import get_model_paths
from .repo_manager import get_all_supported_repo_names, setup
from .utils import logging
from .utils.deps import (
    get_dep_version,
    get_genai_dep_specs,
    get_genai_fastdeploy_spec,
    get_paddle2onnx_dep_specs,
    get_serving_dep_specs,
    is_dep_available,
    is_paddle2onnx_plugin_available,
)
from .utils.env import (
    get_gpu_compute_capability,
    get_paddle_cuda_version,
    is_cuda_available,
)
from .utils.install import install_packages, uninstall_packages
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
        nargs="*",
        metavar="PLUGIN",
        help="Install specified PaddleX plugins.",
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
        help="Use high-performance inference plugin.",
    )
    pipeline_group.add_argument(
        "--hpi_config",
        type=ast.literal_eval,
        help="High-performance inference configuration.",
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
    # Serving also uses `--pipeline`, `--device`, `--use_hpip`, and `--hpi_config`

    ################# paddle2onnx #################
    paddle2onnx_group.add_argument(
        "--paddle2onnx",
        action="store_true",
        help="Convert PaddlePaddle model to ONNX format.",
    )
    paddle2onnx_group.add_argument(
        "--paddle_model_dir",
        type=str,
        help="Directory containing the PaddlePaddle model.",
    )
    paddle2onnx_group.add_argument(
        "--onnx_model_dir",
        type=str,
        help="Output directory for the ONNX model.",
    )
    paddle2onnx_group.add_argument(
        "--opset_version", type=int, default=7, help="Version of the ONNX opset to use."
    )

    # Parse known arguments to get the pipeline name
    args, remaining_args = parser.parse_known_args()
    pipeline = args.pipeline
    pipeline_args = []

    if (
        not (args.install is not None or args.serve or args.paddle2onnx)
        and pipeline is not None
    ):
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
        try:
            install_packages(get_serving_dep_specs())
        except Exception:
            logging.error("Installation failed", exc_info=True)
            sys.exit(1)
        logging.info("Successfully installed the serving plugin")

    def _install_paddle2onnx_deps():
        try:
            install_packages(get_paddle2onnx_dep_specs())
        except Exception:
            logging.error("Installation failed", exc_info=True)
            sys.exit(1)
        logging.info("Successfully installed the Paddle2ONNX plugin")

    def _install_hpi_deps(device_type):
        SUPPORTED_DEVICE_TYPES = ["cpu", "gpu", "npu"]
        if device_type not in SUPPORTED_DEVICE_TYPES:
            logging.error(
                "Failed to install the high-performance plugin.\n"
                "Supported device types: %s. Your input device type: %s.\n",
                SUPPORTED_DEVICE_TYPES,
                device_type,
            )
            sys.exit(2)

        hpip_links_file = "hpip_links.html"
        if device_type == "gpu":
            cuda_version = get_paddle_cuda_version()
            if not cuda_version:
                sys.exit(
                    "No CUDA version found. Please make sure you have installed PaddlePaddle with CUDA enabled."
                )
            if cuda_version[0] == 12:
                hpip_links_file = "hpip_links_cu12.html"
            elif cuda_version[0] != 11:
                sys.exit(
                    "Currently, only CUDA versions 11.x and 12.x are supported by the high-performance inference plugin."
                )

        package_mapping = {
            "cpu": "ultra-infer-python",
            "gpu": "ultra-infer-gpu-python",
            "npu": "ultra-infer-npu-python",
        }
        package = package_mapping[device_type]
        other_packages = set(package_mapping.values()) - {package}
        for other_package in other_packages:
            version = get_dep_version(other_package)
            if version is not None:
                logging.info(
                    f"The high-performance inference plugin '{package}' is mutually exclusive with '{other_package}' (version {version} installed). Uninstalling '{other_package}'..."
                )
                try:
                    uninstall_packages([other_package])
                except Exception:
                    logging.error("Failed to uninstall packages", exc_info=True)
                    sys.exit(1)

        with importlib.resources.path("paddlex", hpip_links_file) as f:
            version = get_dep_version(package)
            try:
                if version is None:
                    install_packages(
                        [package], pip_install_opts=["--find-links", str(f)]
                    )
                else:
                    response = input(
                        f"The high-performance inference plugin is already installed (version {repr(version)}). Do you want to reinstall it? (y/n):"
                    )
                    if response.lower() in ["y", "yes"]:
                        uninstall_packages([package])
                        install_packages(
                            [package],
                            pip_install_opts=[
                                "--find-links",
                                str(f),
                            ],
                        )
                    else:
                        return
            except Exception:
                logging.error("Installation failed", exc_info=True)
                sys.exit(1)

        logging.info("Successfully installed the high-performance inference plugin")

        if not is_paddle2onnx_plugin_available():
            logging.info(
                "The Paddle2ONNX plugin is not available. It is recommended to run `paddlex --install paddle2onnx` to install the Paddle2ONNX plugin to use the full functionality of high-performance inference."
            )

    def _install_genai_deps(plugin_types):
        fd_plugin_types = []
        not_fd_plugin_types = []
        for plugin_type in plugin_types:
            if "fastdeploy" in plugin_type:
                fd_plugin_types.append(plugin_type)
            else:
                not_fd_plugin_types.append(plugin_type)
        if fd_plugin_types:
            if not is_dep_available("paddlepaddle"):
                sys.exit("Please install PaddlePaddle first.")

            cap = get_gpu_compute_capability()
            if cap in ((8, 0), (9, 0)):
                index_url = "https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/"
            elif cap in ((8, 6), (8, 9)):
                index_url = "https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-86_89/"
            else:
                sys.exit(
                    f"The compute capability of the GPU is {cap[0]}.{cap[1]}, which is not supported. The supported compute capabilities are 8.0, 8.6, 8.9, and 9.0."
                )
            try:
                install_packages(
                    [get_genai_fastdeploy_spec("gpu")],
                    pip_install_opts=["--extra-index-url", index_url],
                )
            except Exception:
                logging.error("Installation failed", exc_info=True)
                sys.exit(1)

        reqs = []
        for plugin_type in not_fd_plugin_types:
            try:
                r = get_genai_dep_specs(plugin_type)
            except ValueError:
                logging.error("Invalid generative AI plugin type: %s", plugin_type)
                sys.exit(2)
            reqs += r
        try:
            install_packages(reqs, constraints="required")
        except Exception:
            logging.error("Installation failed", exc_info=True)
            sys.exit(1)

        for plugin_type in plugin_types:
            if "vllm" in plugin_type or "sglang" in plugin_type:
                install_packages(["xformers"], constraints="required")
                if is_cuda_available():
                    try:
                        install_packages(["wheel"], constraints="required")
                        cap = get_gpu_compute_capability()
                        assert cap is not None
                        if cap >= (12, 0):
                            install_packages(
                                ["flash-attn == 2.8.3"], constraints="required"
                            )
                        else:
                            install_packages(
                                ["flash-attn == 2.8.2"], constraints="required"
                            )
                    except Exception:
                        logging.error("Installation failed", exc_info=True)
                        sys.exit(1)
                    break

        logging.info(
            "Successfully installed the generative AI plugin"
            + ("s" if len(plugin_types) > 1 else "")
        )

    # Enable debug info
    os.environ["PADDLE_PDX_DEBUG"] = "True"
    # Disable eager initialization
    os.environ["PADDLE_PDX_EAGER_INIT"] = "False"

    plugins = args.install[:]

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
        for p in hpi_plugins:
            plugins.remove(p)
        if plugins:
            logging.error("`hpi-xxx` cannot be used together with other plugins.")
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

    genai_plugins = list(filter(lambda name: name.startswith("genai-"), plugins))
    if genai_plugins:
        for p in genai_plugins:
            plugins.remove(p)
        if plugins:
            logging.error("`genai-xxx` cannot be used together with other plugins.")
            sys.exit(2)
        genai_plugin_types = [p[len("genai-") :] for p in genai_plugins]
        _install_genai_deps(genai_plugin_types)
        return

    all_repo_names = get_all_supported_repo_names()
    unknown_plugins = []
    for p in plugins:
        if p not in all_repo_names:
            unknown_plugins.append(p)
    if unknown_plugins:
        logging.error("Unknown plugins: %s", unknown_plugins)
        sys.exit(2)
    if plugins:
        repo_names = plugins
    elif len(plugins) == 0:
        repo_names = all_repo_names
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
    hpi_config,
    **pipeline_args,
):
    """pipeline predict"""
    pipeline = create_pipeline(
        pipeline, device=device, use_hpip=use_hpip, hpi_config=hpi_config
    )
    result = pipeline.predict(input, **pipeline_args)
    for res in result:
        res.print()
        if save_path:
            res.save_all(save_path=save_path)


def serve(pipeline, *, device, use_hpip, hpi_config, host, port):
    try:
        from .inference.serving.basic_serving import create_pipeline_app, run_server
    except RuntimeError:
        logging.error("Failed to load the serving module", exc_info=True)
        sys.exit(1)

    pipeline_config = load_pipeline_config(pipeline)
    try:
        pipeline = create_pipeline(
            config=pipeline_config,
            device=device,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
        )
    except Exception:
        logging.error("Failed to create the pipeline", exc_info=True)
        sys.exit(1)
    app = create_pipeline_app(pipeline, pipeline_config)
    run_server(app, host=host, port=port)


# TODO: Move to another module
def paddle_to_onnx(paddle_model_dir, onnx_model_dir, *, opset_version):
    if not is_paddle2onnx_plugin_available():
        sys.exit("Please install the Paddle2ONNX plugin first.")

    ONNX_MODEL_FILENAME = f"{MODEL_FILE_PREFIX}.onnx"
    CONFIG_FILENAME = f"{MODEL_FILE_PREFIX}.yml"
    ADDITIONAL_FILENAMES = ["scaler.pkl"]

    def _check_input_dir(input_dir):
        if input_dir is None:
            sys.exit("Input directory must be specified")
        if not input_dir.exists():
            sys.exit(f"{input_dir} does not exist")
        if not input_dir.is_dir():
            sys.exit(f"{input_dir} is not a directory")
        model_paths = get_model_paths(input_dir)
        if "paddle" not in model_paths:
            sys.exit("PaddlePaddle model does not exist")
        config_path = input_dir / CONFIG_FILENAME
        if not config_path.exists():
            sys.exit(f"{config_path} does not exist")

    def _check_paddle2onnx():
        if shutil.which("paddle2onnx") is None:
            sys.exit("Paddle2ONNX is not available. Please install the plugin first.")

    def _run_paddle2onnx(input_dir, output_dir, opset_version):
        model_paths = get_model_paths(input_dir)
        logging.info("Paddle2ONNX conversion starting...")
        # XXX: To circumvent Paddle2ONNX's bug
        cmd = [
            "paddle2onnx",
            "--model_dir",
            str(model_paths["paddle"][0].parent),
            "--model_filename",
            str(model_paths["paddle"][0].name),
            "--params_filename",
            str(model_paths["paddle"][1].name),
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

    if not paddle_model_dir:
        sys.exit("PaddlePaddle model directory must be specified")

    paddle_model_dir = Path(paddle_model_dir)
    if not onnx_model_dir:
        onnx_model_dir = paddle_model_dir
    onnx_model_dir = Path(onnx_model_dir)
    logging.info(f"Input dir: {paddle_model_dir}")
    logging.info(f"Output dir: {onnx_model_dir}")
    _check_input_dir(paddle_model_dir)
    _check_paddle2onnx()
    _run_paddle2onnx(paddle_model_dir, onnx_model_dir, opset_version)
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

    if args.install is not None:
        install(args)
    elif args.serve:
        serve(
            args.pipeline,
            device=args.device,
            use_hpip=args.use_hpip or None,
            hpi_config=args.hpi_config,
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
            try:
                pipeline_predict(
                    args.pipeline,
                    args.input,
                    args.device,
                    args.save_path,
                    use_hpip=args.use_hpip or None,
                    hpi_config=args.hpi_config,
                    **pipeline_args_dict,
                )
            except Exception:
                logging.error("Pipeline prediction failed", exc_info=True)
                sys.exit(1)
