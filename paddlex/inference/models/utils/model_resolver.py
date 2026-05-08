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

"""Model name, directory, and config resolution."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from .model_config import load_model_config


def resolve_model_name(
    *,
    model_name: str,
    model_dir: Optional[Union[str, Path]] = None,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Path], Dict[str, Any]]:
    """Resolve model name, directory, and config."""
    resolved_dir: Optional[Path] = Path(model_dir) if model_dir else None
    if model_config is not None:
        resolved_config = model_config
    elif resolved_dir is not None:
        resolved_config = load_model_config(resolved_dir)
    else:
        resolved_config = {}
    config_model_name = resolved_config.get("Global", {}).get("model_name")
    if config_model_name and config_model_name != model_name:
        raise ValueError(
            f"Model name mismatch: expected {model_name!r} but config has "
            f"{config_model_name!r}. Please input the correct model dir."
        )
    return model_name, resolved_dir, resolved_config
