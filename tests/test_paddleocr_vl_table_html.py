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

"""Dependency-light regression test for table-image HTML rendering."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace


def test_untokenize_figure_of_table_emits_valid_image_tag():
    source = (
        Path(__file__).resolve().parents[1]
        / "paddlex/inference/pipelines/paddleocr_vl/uilts.py"
    )
    function = next(
        node
        for node in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "untokenize_figure_of_table"
    )
    namespace = {"re": re}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    result = namespace["untokenize_figure_of_table"](
        "<td>[F1]</td>",
        {"[F1]": "imgs/example.jpg"},
        {"imgs/example.jpg": SimpleNamespace(content="")},
    )

    assert result == '<td><img src="imgs/example.jpg" alt="Image" /></td>'
