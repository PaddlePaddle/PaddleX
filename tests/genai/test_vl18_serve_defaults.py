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
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
GENAI = ROOT / "paddlex/inference/genai"
PACKAGE = "paddlex.inference.genai"
SDK = ROOT / "deploy/hps/sdk/pipelines/PaddleOCR-VL-1.8"
NAME = "PaddleOCR-VL-1.8-0.9B"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [1, 2])
def test_nvidia_vl18_calls_native_serve(monkeypatch, count):
    deps = SimpleNamespace(
        is_genai_engine_plugin_available=Mock(return_value=True),
        require_genai_engine_plugin=Mock(),
    )
    configure = Mock()
    for name, value in {
        "paddlex.utils": SimpleNamespace(logging=Mock()),
        "paddlex.utils.deps": deps,
        "torch": SimpleNamespace(version=SimpleNamespace(cuda="12.9", hip=None)),
        f"{PACKAGE}.models": SimpleNamespace(
            ALL_MODEL_INFO={NAME},
            get_model_components=Mock(),
            is_integrated_model_available=Mock(return_value=True),
        ),
        f"{PACKAGE}.backends._vllm_ocr_batching": SimpleNamespace(
            OCR_BATCHED_ARCH="unused",
            OCR_BATCHED_IMPL="unused",
            VISION_ROPE_MODELS={},
            configure_vllm_ocr_vision=configure,
        ),
        f"{PACKAGE}.configs.utils": load(
            f"{PACKAGE}.configs.utils", GENAI / "configs/utils.py"
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    backend = load(f"{PACKAGE}.backends.vllm", GENAI / "backends/vllm.py")
    launch = Mock()
    monkeypatch.setattr(backend.os, "execv", launch)
    config = {"api-server-count": count, "kv-cache-memory-bytes": 4 << 30}
    backend.run_vllm_server("127.0.0.1", 8080, NAME, "/checkpoint", config, None)
    exe, argv = launch.call_args.args
    assert exe == sys.executable
    assert argv[:5] == [exe, "-m", "vllm.entrypoints.cli.main", "serve", "/checkpoint"]
    assert argv[argv.index("--api-server-count") + 1] == str(count)
    assert argv[argv.index("--kv-cache-memory-bytes") + 1] == str(4 << 30)
    assert config == {"api-server-count": count, "kv-cache-memory-bytes": 4 << 30}
    configure.assert_called_once()


@pytest.mark.parametrize(
    "path",
    [
        SDK / "server/pipeline_config.yaml",
        ROOT / "paddlex/configs/pipelines/PaddleOCR-VL-1.8.yaml",
    ],
)
def test_layout_batch_default(path):
    assert (
        yaml.safe_load(path.read_text())["SubModules"]["LayoutDetection"]["batch_size"]
        == 4
    )


@pytest.mark.parametrize("override,expected", [([], 160), (["--concurrency", "2"], 2)])
def test_batch_client_concurrency(monkeypatch, tmp_path, override, expected):
    client = load("vl18_batch_client", SDK / "client/batch_client.py")
    images = tmp_path / "images"
    images.mkdir()
    (images / "page.png").touch()
    request = Mock()
    monkeypatch.setattr(client, "submit_page", request)
    pool = MagicMock()
    pool.return_value.__enter__.return_value.map.side_effect = map
    monkeypatch.setattr(client, "ThreadPoolExecutor", pool)
    assert (
        client.main(
            [
                "--images-dir",
                str(images),
                "--output-dir",
                str(tmp_path / "out"),
                *override,
            ]
        )
        == 0
    )
    pool.assert_called_once_with(max_workers=expected)
    request.assert_called_once()
