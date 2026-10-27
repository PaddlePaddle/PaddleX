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

import importlib.util
import itertools
import sys
import types
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def remove_overlaps():
    # Load the real geometry module without initializing model runtimes. The
    # OCRResult dependency below is used only in unrelated type annotations.
    root = Path(__file__).resolve().parents[1]
    prefix = "paddlex.inference.pipelines"
    modules = {}
    for name in ["paddlex", "paddlex.inference", prefix, prefix + ".layout_parsing"]:
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split(".")))]
        modules[name] = module
    components = types.ModuleType(prefix + ".components")
    conversion = (
        root
        / "paddlex/inference/pipelines/components/common/convert_points_and_boxes.py"
    )
    spec = importlib.util.spec_from_file_location("layout_box_conversion", conversion)
    conversion_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conversion_module)
    components.convert_points_to_boxes = conversion_module.convert_points_to_boxes
    modules[components.__name__] = components
    result = types.ModuleType(prefix + ".ocr.result")
    result.OCRResult = dict
    modules[result.__name__] = result
    with patch.dict(sys.modules, modules):
        name = prefix + ".layout_parsing.utils"
        spec = importlib.util.spec_from_file_location(
            name, root / "paddlex/inference/pipelines/layout_parsing/utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.remove_overlap_blocks


def box(coordinate, score=0.5, label="text"):
    return {"coordinate": coordinate, "score": score, "label": label}


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_retains_higher_confidence_columns_in_any_detection_order(
    remove_overlaps, order
):
    columns = [box([45, 76, 451, 350], 0.552), box([721, 76, 1002, 319], 0.471)]
    detections = columns + [box([45, 74, 1003, 350], 0.402)]
    source = {"boxes": [detections[i] for i in order], "input_path": "two-columns.png"}
    original = deepcopy(source)

    result = remove_overlaps(source, threshold=0.5, smaller=True)

    assert sorted(item["coordinate"] for item in result["boxes"]) == sorted(
        item["coordinate"] for item in columns
    )
    assert result["input_path"] == source["input_path"]
    assert source == original


def test_discards_line_detections_but_keeps_both_columns(remove_overlaps):
    coordinates_and_scores = [
        ([45, 76, 451, 350], 0.552),
        ([47, 324, 396, 349], 0.471),
        ([721, 76, 1002, 319], 0.471),
        ([48, 270, 442, 295], 0.450),
        ([48, 132, 452, 156], 0.445),
        ([47, 77, 295, 104], 0.419),
        ([799, 76, 1000, 102], 0.413),
        ([793, 294, 1000, 318], 0.409),
        ([720, 130, 1001, 156], 0.403),
        ([45, 74, 1003, 350], 0.402),
    ]
    result = remove_overlaps(
        {"boxes": [box(coords, score) for coords, score in coordinates_and_scores]},
        threshold=0.5,
    )
    assert [item["coordinate"] for item in result["boxes"]] == [
        [45, 76, 451, 350],
        [721, 76, 1002, 319],
    ]


@pytest.mark.parametrize(
    "outer,children",
    [
        (box([0, 0, 100, 100], 0.9), [box([0, 0, 40, 100]), box([60, 0, 100, 100])]),
        (box([0, 0, 100, 100]), [box([0, 0, 40, 100]), box([60, 0, 100, 100])]),
        (box([0, 0, 100, 100], 0.2), [box([0, 0, 100, 40]), box([0, 60, 100, 100])]),
        (box([0, 0, 100, 100], 0.2), [box([0, 0, 70, 100]), box([30, 0, 100, 100])]),
        (
            box([0, 0, 100, 100], 0.2, "table"),
            [box([0, 0, 40, 100]), box([60, 0, 100, 100])],
        ),
        (box([0, 0, 100, 100], None), [box([0, 0, 40, 100]), box([60, 0, 100, 100])]),
    ],
)
def test_preserves_existing_container_selection(remove_overlaps, outer, children):
    result = remove_overlaps({"boxes": [outer] + children}, threshold=0.5)
    assert result["boxes"] == [outer]


def test_preserves_larger_removal_mode(remove_overlaps):
    children = [box([0, 0, 40, 100]), box([60, 0, 100, 100])]
    result = remove_overlaps(
        {"boxes": [box([0, 0, 100, 100], 0.2)] + children}, smaller=False
    )
    assert result["boxes"] == children
