# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import weakref
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator
from typing_extensions import Literal

from ....utils.deps import class_requires_deps

SERVER_BACKENDS = ["fastdeploy-server", "vllm-server", "sglang-server"]


class GenAIConfig(BaseModel):
    backend: Literal["native", "fastdeploy-server", "vllm-server", "sglang-server"]
    server_url: Optional[str] = None
    client_kwargs: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def check_server_url(self):
        if self.backend in SERVER_BACKENDS and self.server_url is None:
            raise ValueError(
                f"`server_url` must not be `None` for the {repr(self.backend)} backend."
            )
        return self


def need_local_model(genai_config):
    if genai_config is not None and genai_config.backend in SERVER_BACKENDS:
        return False
    return True


@class_requires_deps("openai")
class GenAIClient(object):
    def __init__(self, backend, base_url, model_name=None, **kwargs):
        from openai import OpenAI

        super().__init__()

        self.backend = backend

        if "api_key" not in kwargs:
            kwargs["api_key"] = "null"
        self._client = OpenAI(base_url=base_url, **kwargs)

        if model_name is not None:
            self._model = model_name
        else:
            try:
                models = self._client.models.list()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to get the model list from the OpenAI-compatible server: {e}"
                ) from e
            self._model = models.data[0].id

        self._finalizer = weakref.finalize(self, self._close, self._client)

    @property
    def openai_client(self):
        return self._client

    def create_chat_completion(self, messages, **kwargs):
        return self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **kwargs,
        )

    def close(self):
        self._close(self._client)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        self.close()

    @staticmethod
    def _close(client):
        client.close()
