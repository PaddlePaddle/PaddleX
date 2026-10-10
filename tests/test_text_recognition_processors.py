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

from paddlex.inference.models.text_recognition.processors import CTCLabelDecode


def _make_logits():
    token_ids = np.array([[1, 1, 0, 2]], dtype=np.int64)
    logits = np.full((1, token_ids.shape[1], 3), -1.0, dtype=np.float32)
    np.put_along_axis(logits, token_ids[..., None], 1.0, axis=-1)
    return logits


def test_ctc_label_decode_does_not_copy_numpy_predictions(monkeypatch):
    logits = _make_logits()
    original_asarray = np.asarray
    conversion = {}

    def record_asarray(value):
        result = original_asarray(value)
        conversion["shares_memory"] = np.shares_memory(result, value)
        return result

    monkeypatch.setattr(np, "asarray", record_asarray)

    decoder = CTCLabelDecode(character_list=["a", "b"])
    texts, scores = decoder([logits])

    assert conversion["shares_memory"]
    assert texts == ["ab"]
    assert len(scores) == 1


def test_ctc_label_decode_keeps_array_like_compatibility():
    logits = _make_logits()
    decoder = CTCLabelDecode(character_list=["a", "b"])

    ndarray_result = decoder([logits])
    list_result = decoder([logits.tolist()])

    assert list_result == ndarray_result
