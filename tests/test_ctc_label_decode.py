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

import ast
import re
from pathlib import Path

import numpy as np
import pytest


def _load_decoder():
    # Test the NumPy decoder without initializing inference backends.
    path = (
        Path(__file__).resolve().parents[1]
        / "paddlex/inference/models/text_recognition/processors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in {"BaseRecLabelDecode", "CTCLabelDecode"}
    ]
    for node in classes:
        node.decorator_list = []
    namespace = {"np": np, "re": re}
    exec(
        compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["CTCLabelDecode"]


CTCLabelDecode = _load_decoder()


def _reference(decoder, logits, return_word_box, **kwargs):
    preds = np.array(logits)
    decoded = decoder.decode(
        preds.argmax(axis=-1),
        preds.max(axis=-1),
        is_remove_duplicate=True,
        return_word_box=return_word_box,
    )
    if return_word_box:
        for index, item in enumerate(decoded):
            item[2][0] *= kwargs["wh_ratio_list"][index] / kwargs["max_wh_ratio"]
    return (
        [item[0] if len(item) <= 2 else (item[0], item[2]) for item in decoded],
        [item[1] for item in decoded],
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "fortran", "list"])
@pytest.mark.parametrize("return_word_box", [False, True])
def test_ctc_matches_max_reduction(dtype, layout, return_word_box):
    logits = np.random.default_rng(7).random((4, 24, 7)).astype(dtype)
    logits[0, 0, 2:4] = 2  # Equal scores retain the first winning character.
    logits[0, 1, :] = 0
    logits[0, 1, 0] = 2  # Blank.
    logits[0, 2:4, 3] = 2  # Repeated character.
    if layout == "strided":
        logits = logits[:, ::2, ::-1]
    elif layout == "fortran":
        logits = np.asfortranarray(logits)
    before = logits.copy()
    pred = logits.tolist() if layout == "list" else logits
    decoder = CTCLabelDecode(character_list=list("a中1-."))
    kwargs = {"wh_ratio_list": [1, 2, 3, 4], "max_wh_ratio": 4}
    actual = decoder([pred], return_word_box=return_word_box, **kwargs)
    expected = _reference(decoder, pred, return_word_box, **kwargs)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(logits, before)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0])
def test_ctc_nonfinite_scores(value):
    logits = np.ones((1, 3, 4), dtype=np.float32)
    logits[0, 0, 2] = value
    logits[0, 1, :] = value
    decoder = CTCLabelDecode(character_list=list("ab"))
    actual = decoder([logits])
    expected = _reference(decoder, logits, False)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])


def test_ctc_all_blank():
    logits = np.zeros((2, 4, 4), dtype=np.float32)
    decoder = CTCLabelDecode(character_list=list("ab"))
    assert decoder([logits]) == (["", ""], [0.0, 0.0])
