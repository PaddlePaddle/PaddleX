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

import numpy as np
import pytest

from paddlex.inference.models.text_recognition.processors import BaseRecLabelDecode
from paddlex.inference.pipelines.components.common.cal_ocr_word_box import (
    cal_ocr_word_box,
)


@pytest.mark.parametrize(
    "text", ["Köln", "Flächennutzung", "Naturverhältnisse", "café", "Łódź"]
)
def test_latin_word_keeps_decoded_columns_and_box(text):
    selection = np.zeros(2 * len(text), dtype=bool)
    selection[::2] = True
    info = BaseRecLabelDecode().get_word_info(text, selection)
    assert info == ([list(text)], [list(range(0, len(selection), 2))], ["en&num"])
    box = np.array([[0, 0], [200, 0], [200, 20], [0, 20]], dtype=float)
    words, boxes = cal_ocr_word_box(text, box, [len(selection), *info])
    assert words == [text]
    end = int((len(selection) - 1) * 200 / len(selection))
    np.testing.assert_array_equal(boxes, [[[0, 0], [end, 0], [end, 20], [0, 20]]])


@pytest.mark.parametrize(
    "text,groups,states",
    [
        ("hello_world", ["hello", "_", "world"], ["en&num", "symbol", "en&num"]),
        ("VGG-16 3.14", ["VGG-16", " ", "3.14"], ["en&num", "symbol", "en&num"]),
        ("你好", ["你好"], ["cn"]),
        ("かな", ["かな"], ["symbol"]),
        ("café!", ["café", "!"], ["en&num", "symbol"]),
    ],
)
def test_existing_separators_and_scripts(text, groups, states):
    words, columns, actual_states = BaseRecLabelDecode().get_word_info(
        text, np.ones(len(text), dtype=bool)
    )
    assert ["".join(word) for word in words] == groups
    assert actual_states == states
    assert [column for group in columns for column in group] == list(range(len(text)))
