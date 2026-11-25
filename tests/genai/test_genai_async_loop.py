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

import ast
import asyncio
import atexit
import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[2] / "paddlex/inference"
RUNTIME_NAMES = {
    "_AsyncThreadManager",
    "_async_thread_manager",
    "_async_thread_manager_lock",
    "get_async_manager",
    "start_aio_loop",
    "run_async",
}


def load_async_runtime():
    tree = ast.parse((ROOT / "models/common/genai.py").read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if names & RUNTIME_NAMES:
            nodes.append(node)
    code = compile(ast.Module(body=nodes, type_ignores=[]), "genai.py", "exec")
    noop = lambda *args, **kwargs: None
    scope = {
        "asyncio": asyncio,
        "atexit": atexit,
        "concurrent": concurrent,
        "threading": threading,
        "Optional": Optional,
        "logging": SimpleNamespace(debug=noop, warning=noop, error=noop),
    }
    exec(code, scope)
    return scope


def test_concurrent_first_requests_share_one_event_loop(monkeypatch):
    runtime = load_async_runtime()
    created = []
    new_event_loop = asyncio.new_event_loop

    def counting_new_event_loop():
        loop = new_event_loop()
        created.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", counting_new_event_loop)
    num_threads = 16
    barrier = threading.Barrier(num_threads)

    async def running_loop():
        await asyncio.sleep(0.01)
        return id(asyncio.get_running_loop())

    def first_request(_):
        barrier.wait()
        return runtime["run_async"](running_loop(), timeout=10)

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            loops = set(pool.map(first_request, range(num_threads)))
    finally:
        runtime["get_async_manager"]().stop()

    # The shared AsyncOpenAI client must only ever run on one event loop.
    assert len(created) == 1
    assert len(loops) == 1
