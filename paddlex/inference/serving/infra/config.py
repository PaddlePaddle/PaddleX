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

from typing import Any, Dict, Final, Mapping, Optional

from pydantic import BaseModel

__all__ = [
    "SERVING_CONFIG_KEY",
    "AppConfig",
    "create_app_config",
]

SERVING_CONFIG_KEY: Final[str] = "Serving"


class AppConfig(BaseModel):
    visualize: bool = True
    extra: Optional[Dict[str, Any]] = None


def create_app_config(pipeline_config: Mapping[str, Any], **kwargs: Any) -> AppConfig:
    app_config = pipeline_config.get(SERVING_CONFIG_KEY, {})
    app_config.update(kwargs)
    return AppConfig.model_validate(app_config)
