# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import paddle

from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import _siglip


class TestChunkedEagerAttention(unittest.TestCase):
    @unittest.skipUnless(
        paddle.is_compiled_with_cuda(), "CUDA required for FP16 matmul"
    )
    def test_fp16_cuda_matches_dense(self):
        paddle.set_device("gpu:0")
        paddle.seed(123)
        q, k, v = [paddle.randn([1, 4, 257, 16]).cast("float16") for _ in range(3)]
        with paddle.no_grad():
            expected, _ = _siglip.eager_attention_forward(
                SimpleNamespace(training=True), q, k, v, None, scaling=0.25
            )
            actual, weights = _siglip.eager_attention_forward(
                SimpleNamespace(training=True),
                q,
                k,
                v,
                None,
                scaling=0.25,
                query_chunk_size=64,
            )
        self.assertIsNone(weights)
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=0.002, atol=0.001
        )

    def test_matches_dense_with_query_and_broadcast_masks(self):
        paddle.set_device("cpu")
        paddle.seed(123)
        q = paddle.randn([2, 3, 257, 16])
        k = paddle.randn([2, 3, 311, 16])
        v = paddle.randn([2, 3, 311, 16])
        module = SimpleNamespace(training=False)
        for shape in (None, (257, 311), (2, 1, 257, 311), (2, 1, 1, 311), (311,)):
            with self.subTest(mask_shape=shape), paddle.no_grad():
                mask = None if shape is None else paddle.randn(shape) * 0.1
                expected, _ = _siglip.eager_attention_forward(
                    module, q, k, v, mask, scaling=0.25
                )
                softmax = _siglip.F.softmax
                query_lengths = []

                def observed(x, *args, **kwargs):
                    query_lengths.append(x.shape[-2])
                    self.assertEqual(x.shape[-1], 311)
                    return softmax(x, *args, **kwargs)

                with patch.object(_siglip.F, "softmax", side_effect=observed):
                    actual, weights = _siglip.eager_attention_forward(
                        module, q, k, v, mask, scaling=0.25, query_chunk_size=64
                    )
                self.assertIsNone(weights)
                self.assertEqual(query_lengths, [64, 64, 64, 64, 1])
                np.testing.assert_allclose(
                    actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6
                )

    def test_training_gradients_and_small_inputs_keep_dense_weights(self):
        paddle.set_device("cpu")
        for training, gradients, length in (
            (True, False, 80),
            (False, True, 80),
            (False, False, 12),
        ):
            with self.subTest(training=training, gradients=gradients, length=length):
                q = paddle.randn([1, 2, length, 8])
                with paddle.set_grad_enabled(gradients):
                    output, weights = _siglip.eager_attention_forward(
                        SimpleNamespace(training=training),
                        q,
                        q,
                        q,
                        None,
                        scaling=0.25,
                        dropout=0.1 if training else 0.0,
                        query_chunk_size=32,
                    )
                self.assertEqual(list(weights.shape), [1, 2, length, length])
                self.assertEqual(list(output.shape), [1, length, 2, 8])

    def test_no_grad_generation_with_training_flag_and_zero_dropout(self):
        paddle.set_device("cpu")
        q = paddle.randn([1, 2, 83, 8])
        module = SimpleNamespace(training=True)
        with paddle.no_grad():
            expected, _ = _siglip.eager_attention_forward(
                module, q, q, q, None, scaling=0.25
            )
            actual, weights = _siglip.eager_attention_forward(
                module,
                q,
                q,
                q,
                None,
                scaling=0.25,
                query_chunk_size=32,
            )
        self.assertIsNone(weights)
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=1e-5, atol=1e-6
        )

    def test_invalid_chunk_size(self):
        q = paddle.zeros([1, 1, 1, 1])
        for size in (0, -1):
            with self.assertRaises(ValueError):
                _siglip.eager_attention_forward(
                    SimpleNamespace(training=False),
                    q,
                    q,
                    q,
                    None,
                    scaling=1.0,
                    query_chunk_size=size,
                )


if __name__ == "__main__":
    unittest.main()
