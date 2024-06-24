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

from . import version
from .modules import build_dataset_checker, build_trainer, build_evaluater, build_predictor
from .modules import create_model, PaddleInferenceOption
from .pipelines import *


def _initialize():
    from .utils.logging import setup_logging
    from .utils import flags
    from . import repo_manager
    from . import repo_apis

    __DIR__ = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    repo_manager.set_parent_dirs(
        os.path.join(__DIR__, 'repo_manager', 'repos'), repo_apis)

    setup_logging()

    if flags.EAGER_INITIALIZATION:
        repo_manager.initialize()


_initialize()

__version__ = version.get_pdx_version()
