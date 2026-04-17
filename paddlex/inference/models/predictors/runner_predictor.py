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

from typing import Any, Dict, Optional

from ..runners import InferenceRunner
from .local_model_predictor import LocalModelPredictor


class RunnerPredictor(LocalModelPredictor):
    """Base class for predictors that use inference runners."""

    __is_base = True

    def __init__(
        self,
        *,
        runner: InferenceRunner,
        model_dir: Optional[str] = None,
        model_config: Optional[Dict] = None,
        model_name: Optional[str] = None,
        engine_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_dir=model_dir,
            model_config=model_config,
            model_name=model_name,
            engine_config=engine_config,
            **kwargs,
        )
        self.runner = runner

    def close(self) -> None:
        close = getattr(self.runner, "close", None)
        if callable(close):
            close()
