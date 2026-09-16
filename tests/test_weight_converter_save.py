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

from paddlex.modules.base.weight_converter import WeightConverter, _preprocess_tensors

safetensors_numpy = pytest.importorskip("safetensors.numpy")


@pytest.mark.parametrize(
    "weight",
    [
        np.arange(21, dtype=np.float32).reshape(3, 7).T,
        np.arange(16, dtype=np.float32).reshape(4, 4).T,
        np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
        np.arange(8, dtype=np.float32)[::-1],
        np.arange(12, dtype=np.float32).reshape(3, 4),
        np.array(7, dtype=np.int64),
        np.int64(0),
    ],
    ids=[
        "linear",
        "square",
        "strided",
        "reversed",
        "contiguous",
        "scalar",
        "numpy-scalar",
    ],
)
def test_save_preserves_values_shape_and_dtype(tmp_path, weight):
    converter = WeightConverter.__new__(WeightConverter)
    converter.output_dir = str(tmp_path)
    original = np.array(weight, copy=True)
    converter._save_safetensors({"weight": weight})
    saved = safetensors_numpy.load_file(tmp_path / "model.safetensors")["weight"]
    assert saved.shape == original.shape
    assert saved.dtype == original.dtype
    np.testing.assert_array_equal(saved, original)
    np.testing.assert_array_equal(weight, original)


def test_save_preprocessed_attention_weights(tmp_path):
    fused = np.arange(48, dtype=np.float32).reshape(4, 12)
    weights = _preprocess_tensors({"attention.in_proj_weight": fused})
    converter = WeightConverter.__new__(WeightConverter)
    converter.output_dir = str(tmp_path)
    converter._save_safetensors(weights)
    saved = safetensors_numpy.load_file(tmp_path / "model.safetensors")
    for index, name in enumerate(("q", "k", "v")):
        np.testing.assert_array_equal(
            saved[f"attention.{name}_proj.weight"],
            fused.T[index * 4 : (index + 1) * 4],
        )
