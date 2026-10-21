#
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
#
"""Regression tests for table markdown HTML escaping."""

import sys
import types
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    return next(
        parent for parent in current.parents if (parent / "pyproject.toml").exists()
    )


def _install_fake_package_tree():
    module_names = [
        "paddlex",
        "paddlex.inference",
        "paddlex.inference.pipelines",
        "paddlex.inference.pipelines.components",
        "paddlex.inference.pipelines.layout_parsing",
        "paddlex.inference.pipelines.layout_parsing.utils",
        "paddlex.inference.pipelines.ocr",
        "paddlex.inference.pipelines.ocr.result",
        "paddlex.inference.pipelines.table_recognition",
        "paddlex.inference.pipelines.table_recognition.result",
    ]
    originals = {name: sys.modules.get(name) for name in module_names}

    paddlex_mod = types.ModuleType("paddlex")
    paddlex_mod.__path__ = []
    inference_mod = types.ModuleType("paddlex.inference")
    inference_mod.__path__ = []
    pipelines_mod = types.ModuleType("paddlex.inference.pipelines")
    pipelines_mod.__path__ = []
    components_mod = types.ModuleType("paddlex.inference.pipelines.components")
    layout_parsing_mod = types.ModuleType("paddlex.inference.pipelines.layout_parsing")
    layout_parsing_mod.__path__ = []
    layout_utils_mod = types.ModuleType(
        "paddlex.inference.pipelines.layout_parsing.utils"
    )
    ocr_mod = types.ModuleType("paddlex.inference.pipelines.ocr")
    ocr_mod.__path__ = []
    ocr_result_mod = types.ModuleType("paddlex.inference.pipelines.ocr.result")
    table_pkg_mod = types.ModuleType("paddlex.inference.pipelines.table_recognition")
    table_pkg_mod.__path__ = []
    table_result_mod = types.ModuleType(
        "paddlex.inference.pipelines.table_recognition.result"
    )

    class OCRResult(dict):
        pass

    class SingleTableRecognitionResult(dict):
        pass

    components_mod.convert_points_to_boxes = lambda *args, **kwargs: None
    layout_utils_mod.get_sub_regions_ocr_res = lambda *args, **kwargs: None
    ocr_result_mod.OCRResult = OCRResult
    table_result_mod.SingleTableRecognitionResult = SingleTableRecognitionResult

    sys.modules["paddlex"] = paddlex_mod
    sys.modules["paddlex.inference"] = inference_mod
    sys.modules["paddlex.inference.pipelines"] = pipelines_mod
    sys.modules["paddlex.inference.pipelines.components"] = components_mod
    sys.modules["paddlex.inference.pipelines.layout_parsing"] = layout_parsing_mod
    sys.modules["paddlex.inference.pipelines.layout_parsing.utils"] = layout_utils_mod
    sys.modules["paddlex.inference.pipelines.ocr"] = ocr_mod
    sys.modules["paddlex.inference.pipelines.ocr.result"] = ocr_result_mod
    sys.modules["paddlex.inference.pipelines.table_recognition"] = table_pkg_mod
    sys.modules["paddlex.inference.pipelines.table_recognition.result"] = (
        table_result_mod
    )

    return originals


def _restore_modules(originals, loaded_name):
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    sys.modules.pop(loaded_name, None)


def _load_table_module(module_filename: str):
    repo_root = _repo_root()
    module_name = (
        f"paddlex.inference.pipelines.table_recognition.{module_filename[:-3]}"
    )
    module_path = (
        repo_root
        / "paddlex"
        / "inference"
        / "pipelines"
        / "table_recognition"
        / module_filename
    )
    originals = _install_fake_package_tree()
    try:
        spec = spec_from_file_location(module_name, module_path)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(originals, module_name)


def _minimal_table_structure():
    return [
        "<html>",
        "<body>",
        "<table>",
        "<tr>",
        "<td></td>",
        "</tr>",
        "</table>",
        "</body>",
        "</html>",
    ]


def _render_v1(ocr_contents):
    module = _load_table_module("table_recognition_post_processing.py")
    return module.get_html_result(
        {0: list(range(len(ocr_contents)))},
        ocr_contents,
        _minimal_table_structure(),
    )


def _render_v2(ocr_contents):
    module = _load_table_module("table_recognition_post_processing_v2.py")
    return module.get_html_result(
        [{0: list(range(len(ocr_contents)))}],
        ocr_contents,
        _minimal_table_structure(),
        [0, 1],
    )


@pytest.mark.parametrize("render_html", [_render_v1, _render_v2])
def test_escapes_html_sensitive_ocr_text_in_single_cell(render_html):
    html = render_html(['<recv response="200" response_txn="invite" />'])

    assert (
        '<td>&lt;recv response=&quot;200&quot; response_txn=&quot;invite&quot; /&gt;</td>'
        in html
    )


@pytest.mark.parametrize("render_html", [_render_v1, _render_v2])
def test_preserves_single_outer_bold_wrapper_when_escaping_cell_text(render_html):
    html = render_html(['<b><pause milliseconds="5000"/></b>'])

    assert '<td><b>&lt;pause milliseconds=&quot;5000&quot;/&gt;</b></td>' in html


@pytest.mark.parametrize("render_html", [_render_v1, _render_v2])
def test_escapes_multi_fragment_bold_cell_text(render_html):
    html = render_html(['<b><recv response="200"', 'response_txn="invite"/></b>'])

    assert (
        '<td><b>&lt;recv response=&quot;200&quot; '
        'response_txn=&quot;invite&quot;/&gt;</b></td>' in html
    )
