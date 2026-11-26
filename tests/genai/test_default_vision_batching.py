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
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Annotated
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
BACKEND = "paddlex.inference.genai.backends._vllm_ocr_batching"
NATIVE = "PaddleOCRVLForConditionalGeneration"


class RaggedInputs:
    pixel_values: Annotated[
        object,
        SimpleNamespace(
            dims=("bn", "p", 3, "patch_size", "patch_size"), dynamic_dims={"p"}
        ),
    ]


class FlatInputs:
    pixel_values: Annotated[
        object,
        SimpleNamespace(dims=("np", 3, "patch_size", "patch_size"), dynamic_dims=set()),
    ]


@pytest.fixture
def batching(monkeypatch, tmp_path):
    utils = ModuleType("paddlex.utils")
    utils.logging = Mock()
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    spec = importlib.util.spec_from_file_location(
        BACKEND, ROOT / "paddlex/inference/genai/backends/_vllm_ocr_batching.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    vllm_model = ModuleType("vllm.model_executor.models.paddleocr_vl")
    vllm_model.PaddleOCRImagePixelInputs = RaggedInputs
    for symbol in (
        "PaddleOCRVLForConditionalGeneration",
        "PaddleOCRVLMultiModalProcessor",
        "PaddleOCRVLProcessingInfo",
        "PaddleOCRVLDummyInputsBuilder",
    ):
        setattr(vllm_model, symbol, object())
    parent = ModuleType("vllm.model_executor.models")
    parent.paddleocr_vl = vllm_model
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.setitem(sys.modules, vllm_model.__name__, vllm_model)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": [NATIVE]}))
    return mod, vllm_model, tmp_path


def test_per_image_contract_enables_batching_without_version_check(batching):
    mod, _, path = batching
    cfg = {"hf_overrides": json.dumps({"some_model_attr": 4})}
    mod.enable_default_vision_batching(cfg, path)
    assert "hf_overrides" not in cfg
    assert json.loads(cfg["hf-overrides"]) == {
        "architectures": [mod.OCR_BATCHED_ARCH],
        "some_model_attr": 4,
    }


@pytest.mark.parametrize("schema", [FlatInputs, object])
def test_native_or_unknown_layout_does_not_select_adapter(batching, schema):
    mod, vllm_model, path = batching
    vllm_model.PaddleOCRImagePixelInputs = schema
    cfg = {}
    mod.enable_default_vision_batching(cfg, path)
    assert cfg == {}


@pytest.mark.parametrize(
    "value",
    [
        '{"architectures":["OtherArchitecture"]}',
        '{"architectures":["PaddleOCRVLForConditionalGeneration"]}',
        "[]",
        "{",
    ],
)
def test_leaves_user_override_untouched(batching, value):
    mod, _, path = batching
    cfg = {"hf-overrides": value}
    mod.enable_default_vision_batching(cfg, path)
    assert cfg == {"hf-overrides": value}


def test_does_not_replace_a_custom_checkpoint_architecture(batching):
    mod, _, path = batching
    (path / "config.json").write_text(json.dumps({"architectures": ["CustomOCR"]}))
    cfg = {}
    mod.enable_default_vision_batching(cfg, path)
    assert cfg == {}


@pytest.mark.parametrize(
    "version,schema,expected",
    [
        ("0.19.0", RaggedInputs, "PaddleXVisionBatchedRoPE"),
        ("0.33.0", FlatInputs, "PaddleXVisionRoPE"),
    ],
)
def test_rope_alias_preserves_batching_owner(
    batching, monkeypatch, version, schema, expected
):
    mod, vllm_model, path = batching
    vllm_model.PaddleOCRImagePixelInputs = schema
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__=version))
    cfg = {mod.VISION_ROPE_OPTION: True, "hf-overrides": '{"another_setting": 3}'}
    mod.configure_vllm_ocr_vision(cfg, "PaddleOCR-VL-1.8-0.9B", path, True)
    assert mod.VISION_ROPE_OPTION not in cfg
    overrides = json.loads(cfg["hf-overrides"])
    assert overrides["architectures"] == [expected + NATIVE]
    assert overrides["another_setting"] == 3
    assert overrides["architectures"][0] in mod.VISION_ROPE_MODELS


@pytest.mark.parametrize("value", [None, "true", 1])
def test_rope_option_requires_yaml_boolean(batching, value):
    mod, _, path = batching
    with pytest.raises(ValueError, match="YAML boolean"):
        mod.configure_vllm_ocr_vision(
            {mod.VISION_ROPE_OPTION: value}, "PaddleOCR-VL-1.8-0.9B", path, True
        )


def test_rope_off_keeps_default_batching(batching):
    mod, _, path = batching
    cfg = {mod.VISION_ROPE_OPTION: False}
    mod.configure_vllm_ocr_vision(cfg, "PaddleOCR-VL-1.8-0.9B", path, True)
    assert json.loads(cfg["hf-overrides"])["architectures"] == [mod.OCR_BATCHED_ARCH]


def test_rope_rejects_untested_runtime_and_schema(batching, monkeypatch):
    mod, model, path = batching
    vllm = SimpleNamespace(__version__="0.21.0")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    with pytest.raises(ValueError, match="0.19.0 or 0.33.0"):
        mod.configure_vllm_ocr_vision(
            {mod.VISION_ROPE_OPTION: True}, "PaddleOCR-VL-1.8-0.9B", path, True
        )
    vllm.__version__ = "0.19.0"
    model.PaddleOCRImagePixelInputs = object
    with pytest.raises(ValueError, match="image schema"):
        mod.configure_vllm_ocr_vision(
            {mod.VISION_ROPE_OPTION: True}, "PaddleOCR-VL-1.8-0.9B", path, True
        )
