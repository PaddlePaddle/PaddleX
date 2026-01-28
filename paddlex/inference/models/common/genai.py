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
from typing import Any, Dict, Optional

from pydantic import BaseModel, model_validator
from typing_extensions import Literal

from ....utils import logging
from ....utils.deps import class_requires_deps

SERVER_BACKENDS = [
    "fastdeploy-server",
    "vllm-server",
    "sglang-server",
    "mlx-vlm-server",
]


class GenAIConfig(BaseModel):
    backend: Literal[
        "native", "fastdeploy-server", "vllm-server", "sglang-server", "mlx-vlm-server"
    ] = "native"
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
    """
    Manages an asyncio event loop running in a dedicated background thread.

    This class provides a bridge between synchronous code and async operations,
    allowing sync code to submit coroutines to be executed in the background
    event loop.

    Thread Safety:
        - Only `run_async()` and `stop()` are designed to be called from other threads
        - All internal asyncio operations are executed within the event loop thread
        - Uses `run_coroutine_threadsafe()` and `call_soon_threadsafe()` for cross-thread
          communication as recommended by Python documentation

    Lifecycle:
        1. `start()` - Creates a daemon thread running `loop.run_forever()`
        2. `run_async(coro)` - Submits coroutines to the loop (returns Future)
        3. `stop(timeout)` - Gracefully shuts down: waits for tasks, cancels remaining,
           then cleans up resources

    Example:
        manager = _AsyncThreadManager()
        manager.start()
        future = manager.run_async(some_async_function())
        result = future.result(timeout=10)
        manager.stop()
    """

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.stopped = False
        self._event_start = threading.Event()
        # Event to signal that cleanup has completed in the loop thread
        self._event_cleanup_done = threading.Event()
        # Flag to reject new tasks during shutdown
        self._shutting_down = False

    def start(self):
        """
        Start the background event loop thread.

        This method is idempotent - calling it multiple times has no effect
        if the loop is already running.

        The method blocks until the event loop is fully initialized and ready
        to accept tasks (synchronized via threading.Event).
        """
        if self.is_running():
            return

        # Reset state for potential restart
        self._shutting_down = False
        self.stopped = False
        self._event_start.clear()
        self._event_cleanup_done.clear()

        def _run_loop():
            # Create a new event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            # Signal that loop is ready to accept tasks
            self._event_start.set()
            try:
                # Run until stop() is called
                self.loop.run_forever()
            finally:
                # IMPORTANT: Cleanup runs in the loop thread after run_forever() returns
                # This ensures all asyncio operations (all_tasks, cancel, etc.) are
                # executed in the correct thread context
                self._cleanup_loop_internal()
                self._event_cleanup_done.set()
                self.stopped = True

        self.thread = threading.Thread(target=_run_loop, daemon=True)
        self.thread.start()
        # Wait for the loop to be fully initialized
        self._event_start.wait()

    def _cleanup_loop_internal(self):
        """
        Perform cleanup operations within the event loop thread.

        IMPORTANT: This method MUST be called from the event loop thread
        (i.e., in the finally block of _run_loop) because asyncio operations
        like `all_tasks()`, `task.cancel()` are NOT thread-safe.

        Cleanup sequence:
        1. Cancel all remaining tasks
        2. Wait for cancellation to complete
        3. Shutdown async generators (prevents ResourceWarning)
        4. Shutdown default executor (Python 3.9+)
        5. Close the event loop
        """
        if self.loop is None:
            return

        try:
            # Get all pending tasks - safe because we're in the loop thread
            pending = asyncio.all_tasks(self.loop)

            if pending:
                logging.debug(f"Cancelling {len(pending)} pending tasks during cleanup")

                # Cancel all tasks
                for task in pending:
                    task.cancel()

                # Run the loop until all cancellations are processed
                # Using gather with return_exceptions=True to collect CancelledError
                self.loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            # Shutdown async generators to prevent ResourceWarning
            # Available since Python 3.6
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())

            # Shutdown the default executor and wait for threads to finish
            # Available since Python 3.9 - check for compatibility with Python 3.8
            if hasattr(self.loop, "shutdown_default_executor"):
                self.loop.run_until_complete(self.loop.shutdown_default_executor())

        except Exception as e:
            logging.warning(f"Error during event loop cleanup: {e}")
        finally:
            self.loop.close()
            logging.debug("Event loop closed successfully")

    def stop(self, timeout: float = 5.0):
        """
        Gracefully stop the event loop.

        This method performs a graceful shutdown sequence:
        1. Sets shutting_down flag to reject new tasks
        2. Schedules a graceful shutdown coroutine in the event loop that:
           - Waits for pending tasks to complete (with timeout)
           - Cancels tasks that don't complete in time
        3. Signals the loop to stop
        4. Waits for cleanup to complete in the loop thread
        5. Joins the background thread

        Args:
            timeout: Maximum seconds to wait for pending tasks to complete.
                    Tasks not completed within this time will be cancelled.
                    Default is 5.0 seconds.

        Thread Safety:
            This method is safe to call from any thread. It uses only thread-safe
            mechanisms (run_coroutine_threadsafe, call_soon_threadsafe, Events)
            to communicate with the event loop thread.
        """
        if not self.is_running():
            return

        # Reject new task submissions
        self._shutting_down = True

        # Define the graceful shutdown coroutine
        # This will be executed IN the event loop thread
        async def _graceful_shutdown():
            """
            Graceful shutdown coroutine that runs inside the event loop.

            All asyncio operations here are thread-safe because this coroutine
            executes in the event loop thread.
            """
            # Get current task to exclude from pending tasks
            current_task = asyncio.current_task()
            pending = [
                t
                for t in asyncio.all_tasks(self.loop)
                if t is not current_task and not t.done()
            ]

            if not pending:
                logging.debug("No pending tasks to wait for during shutdown")
                return

            logging.debug(
                f"Graceful shutdown: waiting for {len(pending)} pending tasks "
                f"(timeout={timeout}s)"
            )

            # Wait for tasks to complete naturally (with timeout)
            done, still_pending = await asyncio.wait(
                pending, timeout=timeout, return_when=asyncio.ALL_COMPLETED
            )

            if still_pending:
                logging.warning(
                    f"Graceful shutdown: cancelling {len(still_pending)} tasks "
                    f"that did not complete within {timeout}s"
                )
                # Cancel tasks that didn't complete in time
                for task in still_pending:
                    task.cancel()

                # Wait for cancellation to be processed
                # return_exceptions=True prevents CancelledError from propagating
                await asyncio.gather(*still_pending, return_exceptions=True)

            logging.debug("Graceful shutdown coroutine completed")

        try:
            # Schedule graceful shutdown in the event loop (thread-safe)
            future = asyncio.run_coroutine_threadsafe(_graceful_shutdown(), self.loop)
            # Wait for graceful shutdown to complete
            # Add extra time for cancellation processing
            future.result(timeout=timeout + 2.0)
        except concurrent.futures.TimeoutError:
            logging.warning(
                f"Graceful shutdown timed out after {timeout + 2.0}s, "
                "forcing loop stop"
            )
        except Exception as e:
            logging.warning(f"Error during graceful shutdown: {e}")

        # Signal the event loop to stop (thread-safe)
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except RuntimeError:
            # Loop may already be closed
            pass

        # Wait for cleanup to complete in the loop thread
        cleanup_completed = self._event_cleanup_done.wait(timeout=5.0)
        if not cleanup_completed:
            logging.warning("Event loop cleanup did not complete within 5s")

        # Wait for the background thread to terminate
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logging.warning(
                    "Background thread did not terminate in time. "
                    "Some resources may not be properly released."
                )

        self.loop = None
        self.thread = None

    def run_async(self, coro):
        """
        Submit a coroutine to be executed in the background event loop.

        This is the primary method for bridging sync and async code.
        The coroutine will be scheduled to run in the background thread's
        event loop.

        Args:
            coro: A coroutine object to be executed

        Returns:
            concurrent.futures.Future: A future that can be used to:
                - Wait for the result: future.result(timeout=...)
                - Check completion: future.done()
                - Cancel the task: future.cancel()

        Raises:
            RuntimeError: If the event loop is not running or is shutting down

        Thread Safety:
            This method is safe to call from any thread. It uses
            `asyncio.run_coroutine_threadsafe()` which is explicitly
            documented as thread-safe.

        Example:
            future = manager.run_async(fetch_data())
            result = future.result(timeout=30)
        """
        if not self.is_running():
            raise RuntimeError("Event loop is not running")

        # Reject new tasks during shutdown to prevent orphaned futures
        if self._shutting_down:
            raise RuntimeError(
                "Event loop is shutting down, cannot accept new tasks. "
                "Please ensure all async operations complete before calling stop()."
            )

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

    def is_running(self):
        """
        Check if the event loop is currently running and accepting tasks.

        Returns:
            bool: True if the loop is running and not closed/stopped
        """
        return self.loop is not None and not self.loop.is_closed() and not self.stopped

    @property
    def is_shutting_down(self):
        """
        Check if the event loop is in the process of shutting down.

        During shutdown, new tasks will be rejected but existing tasks
        are still being processed.

        Returns:
            bool: True if shutdown has been initiated
        """
        return self._shutting_down


_async_thread_manager = None


def get_async_manager():
    global _async_thread_manager
    if _async_thread_manager is None:
        _async_thread_manager = _AsyncThreadManager()
    return _async_thread_manager


def is_aio_loop_ready():
    """
    Check if the async event loop is ready to accept tasks.

    Returns:
        bool: True if the event loop is running and not shutting down
    """
    manager = get_async_manager()
    # Fixed: removed call to non-existent is_closed() method
    return manager.is_running() and not manager.is_shutting_down


def start_aio_loop():
    """
    Start the global async event loop if not already running.

    This function also registers an atexit handler to ensure graceful
    shutdown when the program exits.

    Note:
        The atexit handler calls stop() which performs graceful shutdown,
        waiting for pending tasks to complete before terminating.
    """
    manager = get_async_manager()
    if not manager.is_running():
        manager.start()
        # Register graceful shutdown on program exit
        atexit.register(manager.stop)


def close_aio_loop(timeout: float = 5.0):
    """
    Gracefully close the global async event loop.

    This function initiates a graceful shutdown sequence that:
    1. Waits for pending tasks to complete (up to timeout)
    2. Cancels any remaining tasks
    3. Cleans up resources (async generators, executor)
    4. Closes the event loop

    Args:
        timeout: Maximum seconds to wait for pending tasks to complete.
                Default is 5.0 seconds.
    """
    manager = get_async_manager()
    if manager.is_running():
        manager.stop(timeout=timeout)


def run_async(coro, return_future=False, timeout=None):
    """
    Execute a coroutine in the background event loop.

    This is the main entry point for running async code from sync contexts.
    It automatically starts the event loop if not already running.

    Args:
        coro: The coroutine to execute
        return_future: If True, return a Future immediately without waiting.
                      If False (default), block until the coroutine completes.
        timeout: Maximum seconds to wait for completion (only used when
                return_future=False). None means wait indefinitely.

    Returns:
        If return_future=True: concurrent.futures.Future
        If return_future=False: The result of the coroutine

    Raises:
        RuntimeError: If the event loop fails to start or is shutting down
        concurrent.futures.TimeoutError: If timeout is exceeded
        Exception: Any exception raised by the coroutine

    Example:
        # Blocking call
        result = run_async(fetch_data(), timeout=30)

        # Non-blocking call
        future = run_async(fetch_data(), return_future=True)
        # ... do other work ...
        result = future.result()
    """
    manager = get_async_manager()

    if not manager.is_running():
        start_aio_loop()
        # Note: Removed unnecessary time.sleep(0.1)
        # The start() method already synchronizes via threading.Event,
        # ensuring the loop is fully ready before returning

    if not manager.is_running():
        raise RuntimeError("Failed to start event loop")

    # Check if loop is shutting down (will raise RuntimeError if so)
    if manager.is_shutting_down:
        raise RuntimeError("Event loop is shutting down, cannot accept new tasks")

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
