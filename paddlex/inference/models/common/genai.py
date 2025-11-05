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

import asyncio
import atexit
import concurrent.futures
import threading
import time
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator
from typing_extensions import Literal

from ....utils import logging
from ....utils.deps import class_requires_deps

SERVER_BACKENDS = ["fastdeploy-server", "vllm-server", "sglang-server"]


class GenAIConfig(BaseModel):
    backend: Literal["native", "fastdeploy-server", "vllm-server", "sglang-server"] = (
        "native"
    )
    server_url: Optional[str] = None
    max_concurrency: int = 200
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


# TODO: Can we set the event loop externally?
class _AsyncThreadManager:
    def __init__(self):
        self.loop = None
        self.thread = None
        self.stopped = False
        self._event_start = threading.Event()

    def start(self):
        if self.is_running():
            return

        def _run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self._event_start.set()
            try:
                self.loop.run_forever()
            finally:
                self.loop.close()
                self.stopped = True

        self.thread = threading.Thread(target=_run_loop, daemon=True)
        self.thread.start()
        self._event_start.wait()

    def stop(self):
        # TODO: Graceful shutdown
        if not self.is_running():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=1)
        if self.thread.is_alive():
            logging.warning("Background thread did not terminate in time")
        self.loop = None
        self.thread = None

    def run_async(self, coro):
        if not self.is_running():
            raise RuntimeError("Event loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

    def is_running(self):
        return self.loop is not None and not self.loop.is_closed() and not self.stopped


_async_thread_manager = None


def get_async_manager():
    global _async_thread_manager
    if _async_thread_manager is None:
        _async_thread_manager = _AsyncThreadManager()
    return _async_thread_manager


def is_aio_loop_ready():
    manager = get_async_manager()
    return manager.is_running() and not manager.is_closed()


def start_aio_loop():
    manager = get_async_manager()
    if not manager.is_running():
        manager.start()
        atexit.register(manager.stop)


def close_aio_loop():
    manager = get_async_manager()
    if manager.is_running():
        manager.stop()


def run_async(coro, return_future=False, timeout=None):
    manager = get_async_manager()

    if not manager.is_running():
        start_aio_loop()
        time.sleep(0.1)

    if not manager.is_running():
        raise RuntimeError("Failed to start event loop")

    future = manager.run_async(coro)

    if return_future:
        return future

    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logging.warning(f"Task timed out after {timeout} seconds")
        raise
    except Exception as e:
        logging.error(f"Task failed with error: {e}")
        raise


@class_requires_deps("openai")
class GenAIClient(object):

    def __init__(
        self, backend, base_url, max_concurrency=200, model_name=None, **kwargs
    ):
        from openai import AsyncOpenAI

        super().__init__()

        self.backend = backend
        self._max_concurrency = max_concurrency
        if model_name is None:
            model_name = run_async(self._get_model_name(), timeout=10)
        self._model_name = model_name

        if "api_key" not in kwargs:
            kwargs["api_key"] = "null"
        self._client = AsyncOpenAI(base_url=base_url, **kwargs)

        self._semaphore = asyncio.Semaphore(self._max_concurrency)

    @property
    def openai_client(self):
        return self._client

    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        async def _create_chat_completion_with_semaphore(*args, **kwargs):
            async with self._semaphore:
                return await self._client.chat.completions.create(
                    *args,
                    **kwargs,
                )

        return run_async(
            _create_chat_completion_with_semaphore(
                model=self._model_name,
                messages=messages,
                **kwargs,
            ),
            return_future=return_future,
        )

    def close(self):
        run_async(self._client.close(), timeout=5)

    async def _get_model_name(self):
        try:
            models = await self._client.models.list()
        except Exception as e:
            raise RuntimeError(
                f"Failed to get the model list from the OpenAI-compatible server: {e}"
            ) from e
        return models.data[0].id
