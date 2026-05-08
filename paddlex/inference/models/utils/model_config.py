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

"""Model config loading for inference model directory convention."""

from pathlib import Path
from typing import Any, Dict

from .... import constants
from ...utils.io import YAMLReader


def get_model_config_path(model_dir: Path) -> Path:
    """Return the config file path for an inference model directory."""
    return model_dir / f"{constants.MODEL_FILE_PREFIX}.yml"


def load_model_config(model_dir: Path) -> Dict[str, Any]:
    """Load model config from an inference model directory."""
    return YAMLReader().read(get_model_config_path(model_dir))
