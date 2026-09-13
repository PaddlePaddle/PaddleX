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

from types import SimpleNamespace

import numpy as np
import pytest
from paddlex.inference.pipelines.table_recognition.pipeline_v2 import (
    _TableRecognitionPipelineV2,
)


@pytest.mark.parametrize(
    "cell",
    [
        [5.0, 0.0, 5.0, 10.0],
        [0.0, 5.0, 10.0, 5.0],
        [5.0, 5.0, 5.0, 5.0],
        [10.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
        [20.0, 0.0, 30.0, 10.0],
    ],
)
def test_cell_overlap_preserves_unsplit_ocr(cell):
    result = {"rec_boxes": [[0.0, 0.0, 10.0, 10.0]], "rec_texts": ["original"]}
    actual = _TableRecognitionPipelineV2.split_ocr_bboxes_by_table_cells(
        None, [cell], result, np.zeros((10, 10, 3), dtype=np.uint8)
    )
    assert actual["rec_boxes"] == [[0.0, 0.0, 10.0, 10.0]]
    assert actual["rec_texts"] == ["original"]


def test_zero_area_cell_does_not_prevent_valid_splits():
    crops = []

    def recognize(crop):
        crops.append(crop.shape)
        yield {"rec_text": f"part{len(crops)}"}

    pipeline = SimpleNamespace(
        general_ocr_pipeline=SimpleNamespace(text_rec_model=recognize)
    )
    cells = [[0.0, 0.0, 5.0, 10.0], [5.0, 0.0, 5.0, 10.0], [5.0, 0.0, 10.0, 10.0]]
    result = {"rec_boxes": [[0.0, 0.0, 10.0, 10.0]], "rec_texts": ["original"]}
    actual = _TableRecognitionPipelineV2.split_ocr_bboxes_by_table_cells(
        pipeline, cells, result, np.zeros((10, 10, 3), dtype=np.uint8)
    )
    assert actual["rec_boxes"] == [[0.0, 0.0, 5.0, 10.0], [5.0, 0.0, 10.0, 10.0]]
    assert actual["rec_texts"] == ["part1", "part2"]
    assert crops == [(10, 5, 3), (10, 5, 3)]
