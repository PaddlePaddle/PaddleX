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

import os
import sys

_SPECIAL_MODS = ["paddle", "paddle_custom_device", "ultra_infer"]
_loaded_special_mods = []
for mod in _SPECIAL_MODS:
    if mod in sys.modules:
        _loaded_special_mods.append(mod)

from . import version
from .inference import create_pipeline, create_predictor
from .model import create_model
from .modules import build_dataset_checker, build_evaluator, build_trainer


def _initialize():
    from . import repo_apis, repo_manager
    from .utils import flags
    from .utils.logging import setup_logging

    __DIR__ = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    repo_manager.set_parent_dirs(
        os.path.join(__DIR__, "repo_manager", "repos"), repo_apis
    )

    setup_logging()

    if flags.EAGER_INITIALIZATION:
        repo_manager.initialize()


__version__ = version.get_pdx_version()


_initialize()

for mod in _SPECIAL_MODS:
    if mod in sys.modules and mod not in _loaded_special_mods:
        raise AssertionError(
            f"`{mod}` is unexpectedly loaded. Please contact the PaddleX team to report this issue."
        )
