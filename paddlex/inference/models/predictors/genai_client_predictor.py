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

"""GenAIClientPredictor: base predictor for remote GenAI inference via GenAIClient."""

from typing import Any, Dict, Optional

from ....utils.deps import require_genai_client_plugin
from ..common.genai import SERVER_BACKENDS, GenAIClient, GenAIConfig
from .base_predictor import BasePredictor


class GenAIClientPredictor(BasePredictor):
    """
    Base class for predictors that use remote GenAI services via GenAIClient.
    No local model; delegates inference to remote server.
    """

    __is_base = True

    def __init__(
        self,
        model_name: str,
        engine_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> None:
        require_genai_client_plugin()
        if engine_config is None or not engine_config:
            raise ValueError("GenAIClientPredictor requires `engine_config`.")
        cfg = GenAIConfig.model_validate(engine_config)
        if cfg.backend not in SERVER_BACKENDS:
            raise ValueError(
                f"GenAIClientPredictor requires backend in {SERVER_BACKENDS!r}, "
                f"got {cfg.backend!r}."
            )
        self._genai_config = cfg
        super().__init__(
            model_name=model_name,
            engine_config=engine_config,
            batch_size=batch_size,
            **kwargs,
        )
        client_kwargs = {"model_name": self.model_name}
        client_kwargs.update(self._genai_config.client_kwargs or {})
        self._genai_client = GenAIClient(
            backend=cfg.backend,
            base_url=cfg.server_url,
            max_concurrency=cfg.max_concurrency,
            **client_kwargs,
        )

    @property
    def supports_benchmark(self) -> bool:
        return False

    @property
    def genai_client(self):
        """The underlying `GenAIClient` instance."""
        return self._genai_client

    def close(self) -> None:
        """Close the underlying `GenAIClient` instance."""
        if hasattr(self, "_genai_client") and self._genai_client is not None:
            self._genai_client.close()
