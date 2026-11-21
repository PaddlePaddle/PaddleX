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

"""The background event loop of the GenAI client under concurrent first calls.

Each test runs its scenario in a fresh interpreter: the loop is a process-wide
singleton, and the race only exists while no loop has been started yet.
"""

import subprocess
import sys
import textwrap

import pytest

THREADS = 16


def _run(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_concurrent_first_calls_share_one_event_loop():
    last = _run(
        f"""
        import asyncio, threading
        from paddlex.inference.models.common import genai

        barrier = threading.Barrier({THREADS})
        loops = set()
        lock = threading.Lock()

        async def where():
            return id(asyncio.get_running_loop())

        def worker():
            barrier.wait()
            loop_id = genai.run_async(where(), timeout=30)
            with lock:
                loops.add(loop_id)

        threads = [threading.Thread(target=worker) for _ in range({THREADS})]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(len(loops))
        """
    )
    assert last == "1"


def test_shared_semaphore_survives_concurrent_first_calls():
    # What GenAIClient does: one semaphore, calls from many threads, the
    # semaphore contended right away.
    last = _run(
        f"""
        import asyncio, threading
        from paddlex.inference.models.common import genai

        barrier = threading.Barrier({THREADS})
        semaphore = asyncio.Semaphore(1)
        errors = []

        async def guarded():
            async with semaphore:
                await asyncio.sleep(0.01)

        def worker():
            barrier.wait()
            for _ in range(5):
                try:
                    genai.run_async(guarded(), timeout=30)
                except Exception as error:
                    errors.append(repr(error))

        threads = [threading.Thread(target=worker) for _ in range({THREADS})]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        print(len(errors), errors[:1])
        """
    )
    assert last.startswith("0 "), last


def test_client_semaphore_follows_a_restarted_loop():
    pytest.importorskip("openai")
    last = _run(
        """
        import asyncio
        from paddlex.inference.models.common import genai

        client = genai.GenAIClient(
            backend="vllm-server",
            base_url="http://127.0.0.1:9/v1",
            max_concurrency=1,
            model_name="model",
        )

        async def hold_semaphore():
            semaphore = client._get_semaphore()
            async with semaphore:
                waiter = asyncio.ensure_future(semaphore.acquire())
                await asyncio.sleep(0)
                waiter.cancel()
            return id(semaphore), id(asyncio.get_running_loop())

        first = genai.run_async(hold_semaphore(), timeout=30)
        genai.close_aio_loop()
        second = genai.run_async(hold_semaphore(), timeout=30)
        print(first[1] != second[1], first[0] != second[0])
        """
    )
    assert last == "True True"
