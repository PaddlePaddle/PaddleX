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


import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "paddlex/inference/genai/configs"
PACKAGE = "paddlex.inference.genai.configs"


def load_config_module(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", CONFIG / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cuda,xpu", [("12.9", False), (None, True), (None, False)])
def test_vl18_changes_only_nvidia_vllm_memory_defaults(monkeypatch, cuda, xpu):
    deps = ModuleType("paddlex.utils.deps")
    deps.require_deps = Mock()
    monkeypatch.setitem(sys.modules, "paddlex.utils.deps", deps)
    torch = SimpleNamespace(
        version=SimpleNamespace(cuda=cuda),
        xpu=SimpleNamespace(is_available=lambda: xpu),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    base = load_config_module("paddleocr_vl_09b")
    monkeypatch.setitem(sys.modules, f"{PACKAGE}.paddleocr_vl_09b", base)
    config = load_config_module("paddleocr_vl_18_09b")

    expected = base.get_config("vllm")
    if cuda:
        expected.update(
            {
                "kv-cache-memory-bytes": 4 << 30,
                "max-num-batched-tokens": 32768,
            }
        )
    assert config.get_config("vllm") == expected
    assert config.get_config("sglang") == base.get_config("sglang")


@pytest.mark.parametrize(
    "overrides,expected_cache",
    [
        ({}, 4 << 30),
        ({"kv-cache-memory-bytes": 8 << 30}, 8 << 30),
        ({"kv_cache_memory_bytes": 2 << 30}, 2 << 30),
        ({"gpu_memory_utilization": 0.35}, None),
        ({"kv-cache-memory-bytes": None}, None),
        ({"gpu-memory-utilization": 0.5, "kv_cache_memory_bytes": 8 << 30}, 8 << 30),
    ],
)
def test_explicit_vllm_cache_policy_overrides_default(overrides, expected_cache):
    utils = load_config_module("utils")
    config = utils.merge_vllm_backend_config(
        {"kv-cache-memory-bytes": 4 << 30, "gpu-memory-utilization": 0.5}, overrides
    )
    assert config.get("kv-cache-memory-bytes") == expected_cache
    assert ("--kv-cache-memory-bytes" in utils.backend_config_to_args(config)) == (
        expected_cache is not None
    )
    assert "gpu_memory_utilization" not in config


def test_explicit_vllm_token_budget_overrides_default():
    utils = load_config_module("utils")
    config = utils.merge_vllm_backend_config(
        {"max-num-batched-tokens": 32768}, {"max_num_batched_tokens": 65536}
    )
    assert utils.backend_config_to_args(config) == ["--max-num-batched-tokens", "65536"]
