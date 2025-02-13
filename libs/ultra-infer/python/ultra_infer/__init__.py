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

from __future__ import absolute_import
import logging
import os
import sys
import platform

# Create a symbol link to tensorrt library.
trt_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "libs/third_libs/tensorrt/lib/"
)
if os.name != "nt" and os.path.exists(trt_directory):
    logging.basicConfig(level=logging.INFO)
    for trt_lib in [
        "libnvcaffe_parser.so",
        "libnvinfer_plugin.so",
        "libnvinfer.so",
        "libnvonnxparser.so",
        "libnvparsers.so",
    ]:
        dst = os.path.join(trt_directory, trt_lib)
        src = os.path.join(trt_directory, trt_lib + ".8")
        if not os.path.exists(dst):
            try:
                os.symlink(src, dst)
                logging.info(f"Create a symbolic link pointing to {src} named {dst}.")
            except OSError as e:
                logging.warning(
                    f"Failed to create a symbolic link pointing to {src} by an unprivileged user. "
                    "It may failed when you use Paddle TensorRT backend. "
                    "Please use administator privilege to import ultra_infer at first time."
                )
                break

    # HACK: Reset the root logger config that got messed up by FD.
    root_logger = logging.getLogger()
    root_logger.level = logging.WARNING
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

from .code_version import version, git_version, extra_version_info
from .code_version import enable_trt_backend, enable_paddle_backend, with_gpu

# Note(zhoushunjie): Fix the import order of paddle and ultra_infer library.
# This solution will be removed it when the confilct of paddle and
# ultra_infer is fixed.

# Note(qiuyanjun): Add backward compatible for paddle 2.4.x
sys_platform = platform.platform().lower()


def get_paddle_version():
    paddle_version = ""
    try:
        import pkg_resources

        paddle_version = pkg_resources.require("paddlepaddle-gpu")[0].version.split(
            ".post"
        )[0]
    except:
        try:
            paddle_version = pkg_resources.require("paddlepaddle")[0].version.split(
                ".post"
            )[0]
        except:
            pass
    return paddle_version


def should_import_paddle():
    if ("paddle2.4" in extra_version_info) or ("post24" in extra_version_info):
        paddle_version = get_paddle_version()
        if (
            paddle_version != ""
            and paddle_version <= "2.4.2"
            and paddle_version != "0.0.0"
        ):
            return True
    return False


def should_set_tensorrt():
    if (
        with_gpu == "ON"
        and enable_paddle_backend == "ON"
        and enable_trt_backend == "ON"
    ):
        return True
    return False


def tensorrt_is_avaliable():
    # Note(qiuyanjun): Only support linux now.
    found_trt_lib = False
    if ("linux" in sys_platform) and ("LD_LIBRARY_PATH" in os.environ.keys()):
        for lib_path in os.environ["LD_LIBRARY_PATH"].split(":"):
            if os.path.exists(os.path.join(lib_path, "libnvinfer.so")):
                found_trt_lib = True
                break
    return found_trt_lib


try:
    # windows: no conflict between ultra_infer and paddle.
    # linux: must import paddle first to solve the conflict.
    # macos: still can not solve the conflict between ultra_infer and paddle,
    #        due to the global flags redefined in paddle/paddle_inference so.
    #        we got the error (ERROR: flag 'xxx' was defined more than once).
    if "linux" in sys_platform:
        if should_import_paddle():
            import paddle  # need import paddle first for paddle2.4.x

            # check whether tensorrt in LD_LIBRARY_PATH for ultra_infer
            if should_set_tensorrt() and (not tensorrt_is_avaliable()):
                if os.path.exists(trt_directory):
                    logging.info(
                        "\n[WARNING] Can not find TensorRT lib in LD_LIBRARY_PATH for UltraInfer! \
            \n[WARNING] Please export [ YOUR CUSTOM TensorRT ] lib path to LD_LIBRARY_PATH first, or run the command: \
            \n[WARNING] Linux: 'export LD_LIBRARY_PATH=$(python -c 'from ultra_infer import trt_directory; print(trt_directory)'):$LD_LIBRARY_PATH'"
                    )
                else:
                    logging.info(
                        "\n[WARNING] Can not find TensorRT lib in LD_LIBRARY_PATH for UltraInfer! \
            \n[WARNING] Please export [YOUR CUSTOM TensorRT] lib path to LD_LIBRARY_PATH first."
                    )
except:
    pass


from .c_lib_wrap import (
    ModelFormat,
    Backend,
    FDDataType,
    TensorInfo,
    Device,
    is_built_with_gpu,
    is_built_with_ort,
    ModelFormat,
    is_built_with_paddle,
    is_built_with_trt,
    get_default_cuda_directory,
)


def set_logger(enable_info=True, enable_warning=True):
    """Set behaviour of logger while using UltraInfer

    :param enable_info: (boolean)Whether to print out log level of INFO
    :param enable_warning: (boolean)Whether to print out log level of WARNING, recommend to set to True
    """
    from .c_lib_wrap import set_logger

    set_logger(enable_info, enable_warning)


from .runtime import Runtime, RuntimeOption
from .model import UltraInferModel
from . import c_lib_wrap as C
from . import vision
from . import pipeline
from . import text
from . import ts
from .download import download, download_and_decompress, download_model, get_model_list


__version__ = version
