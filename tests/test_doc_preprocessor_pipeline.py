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

from paddlex.inference.pipelines.doc_preprocessor.pipeline import (
    _to_bgr_contiguous,
)


@pytest.mark.parametrize("shape", [(2, 3, 3), (7, 5, 3)])
def test_to_bgr_contiguous_preserves_pixels(shape):
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)

    output = _to_bgr_contiguous(image)

    np.testing.assert_array_equal(output, image[:, :, ::-1])
    assert output.flags.c_contiguous
    assert all(stride >= 0 for stride in output.strides)


@pytest.mark.parametrize(
    "image",
    [
        np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)[::2, ::2],
        np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3).transpose(1, 0, 2),
    ],
)
def test_to_bgr_contiguous_handles_noncontiguous_input(image):
    expected = image[:, :, ::-1].copy()
    image.setflags(write=False)

    output = _to_bgr_contiguous(image)

    np.testing.assert_array_equal(output, expected)
    assert output.flags.c_contiguous
    assert output.flags.writeable
    assert all(stride >= 0 for stride in output.strides)
