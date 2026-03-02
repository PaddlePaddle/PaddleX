# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Shared utility functions for predictors."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def resolve_model_args(
    model_dir: Optional[str],
    model_config: Optional[Dict],
    model_name: Optional[str],
) -> Tuple[Optional[Path], Dict[str, Any], str]:
    """Resolve common local-model init arguments.

    Returns:
        (resolved_dir, config, resolved_name)
    """
    resolved_dir = Path(model_dir) if model_dir else None
    config = model_config or {}
    resolved_name = model_name or config.get("Global", {}).get("model_name", "")
    return resolved_dir, config, resolved_name
