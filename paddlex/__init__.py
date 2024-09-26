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

from .utils.lazy_loader import LazyLoader
import sys

paddle = LazyLoader("lazy_paddle", globals(), "paddle")
sys.modules["lazy_paddle"] = paddle

import os

from . import version
from .modules import (
    build_dataset_checker,
    build_trainer,
    build_evaluater,
)


from .model import create_model
from .inference import create_predictor, create_pipeline


def _initialize():
    from .utils.logging import setup_logging
    from .utils import flags
    from . import repo_manager
    from . import repo_apis

    __DIR__ = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    repo_manager.set_parent_dirs(
        os.path.join(__DIR__, "repo_manager", "repos"), repo_apis
    )

    setup_logging()

    if flags.EAGER_INITIALIZATION:
        repo_manager.initialize()


def _check_paddle_version():
    """check paddle version"""

    supported_versions = ["3.0", "0.0"]
    device_type = paddle.device.get_device().split(":")[0]
    if device_type.lower() == "xpu":
        supported_versions.append("2.6")
    version = paddle.__version__
    # Recognizable version number: major.minor.patch
    major, minor, patch = version.split(".")
    # Ignore patch
    version = f"{major}.{minor}"
    if version not in supported_versions:
        raise RuntimeError(
            f"The {version} version of PaddlePaddle is not supported. "
            f"Please install one of the following versions of PaddlePaddle: {supported_versions}."
        )


_initialize()

__version__ = version.get_pdx_version()
