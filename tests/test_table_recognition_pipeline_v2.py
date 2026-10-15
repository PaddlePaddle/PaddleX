# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_V2_PATH = (
    REPO_ROOT
    / "paddlex"
    / "inference"
    / "pipelines"
    / "table_recognition"
    / "pipeline_v2.py"
)
PARENT_METHOD_NAME = "split_ocr_bboxes_by_table_cells"
NESTED_FUNCTION_NAME = "calculate_iou"


def _load_calculate_iou():
    """Extract and compile the nested calculate_iou without importing PaddleX.

    Importing the pipeline requires optional runtime and model dependencies, so
    the production source is parsed with the ast module and only the nested
    helper is executed.
    """
    source = PIPELINE_V2_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PIPELINE_V2_PATH))

    parent_methods = [
        node
        for class_node in ast.walk(tree)
        if isinstance(class_node, ast.ClassDef)
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == PARENT_METHOD_NAME
    ]
    if len(parent_methods) != 1:
        raise AssertionError(
            "Expected exactly one {!r} method in {}, found {}".format(
                PARENT_METHOD_NAME, PIPELINE_V2_PATH, len(parent_methods)
            )
        )

    nested_functions = [
        node
        for node in ast.walk(parent_methods[0])
        if isinstance(node, ast.FunctionDef) and node.name == NESTED_FUNCTION_NAME
    ]
    if len(nested_functions) != 1:
        raise AssertionError(
            "Expected exactly one nested {!r} function in {!r}, found {}".format(
                NESTED_FUNCTION_NAME, PARENT_METHOD_NAME, len(nested_functions)
            )
        )

    module = ast.Module(body=[nested_functions[0]], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(PIPELINE_V2_PATH), "exec"), namespace)
    return namespace[NESTED_FUNCTION_NAME]


class SplitOcrBboxesCalculateIouTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calculate_iou = staticmethod(_load_calculate_iou())

    def test_positive_area_cell_returns_overlap_ratio(self):
        ocr_box = [0, 0, 10, 10]
        cell_box = [5, 0, 15, 10]

        self.assertAlmostEqual(self.calculate_iou(ocr_box, cell_box), 0.5)

    def test_zero_width_cell_returns_zero(self):
        ocr_box = [0, 0, 10, 10]
        cell_box = [5, 0, 5, 10]

        self.assertEqual(self.calculate_iou(ocr_box, cell_box), 0.0)

    def test_zero_height_cell_returns_zero(self):
        ocr_box = [0, 0, 10, 10]
        cell_box = [0, 5, 10, 5]

        self.assertEqual(self.calculate_iou(ocr_box, cell_box), 0.0)


if __name__ == "__main__":
    unittest.main()
