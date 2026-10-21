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

import unittest

import numpy as np

from paddlex.inference.models.text_recognition.predictor import (
    TextRecTransformersPredictor,
    _word_box_width_ratios,
)
from paddlex.inference.models.text_recognition.processors import OCRReisizeNormImg


class TestTextRecognitionWidth(unittest.TestCase):
    def test_invalid_options(self):
        for scale in (0, -1, float("nan"), float("inf"), True, "2"):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                OCRReisizeNormImg(width_scale=scale)
        for limit in (0, -1, 0.5, float("inf"), True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                OCRReisizeNormImg(width_limit=limit)

    def test_scaled_crop_uses_cap_and_padding(self):
        crop = np.full((48, 400, 3), 255, dtype=np.uint8)
        default = OCRReisizeNormImg()(imgs=[crop])[0]
        wider = OCRReisizeNormImg(width_scale=2.0, width_limit=640)(imgs=[crop])[0]
        self.assertEqual(default.shape, (3, 48, 400))
        self.assertEqual(wider.shape, (3, 48, 640))

        short = np.full((48, 100, 3), 255, dtype=np.uint8)
        padded = OCRReisizeNormImg(width_scale=2.0)(imgs=[short])[0]
        self.assertEqual(padded.shape, (3, 48, 320))
        self.assertTrue(np.all(padded[:, :, :200] == 1))
        self.assertTrue(np.all(padded[:, :, 200:] == 0))

    def test_word_boxes_use_actual_cap_for_batched_crops(self):
        crops = [np.zeros((48, w, 3), dtype=np.uint8) for w in (720, 960)]
        ratios, padded_ratio = _word_box_width_ratios(
            crops, [0, 1], [3, 48, 320], 2, 2.0, 1280, None
        )
        self.assertEqual(ratios, [1280 / 48, 1280 / 48])
        self.assertEqual(padded_ratio, 1280 / 48)

    def test_static_shape_ignores_width_options(self):
        crop = np.full((48, 720, 3), 255, dtype=np.uint8)
        default = OCRReisizeNormImg(input_shape=[3, 48, 320])([crop])[0]
        wider = OCRReisizeNormImg(
            input_shape=[3, 48, 320], width_scale=2.0, width_limit=1280
        )([crop])[0]
        np.testing.assert_array_equal(default, wider)
        ratios, padded_ratio = _word_box_width_ratios(
            [crop], [0], [3, 48, 320], 1, 2.0, 1280, [3, 48, 320]
        )
        self.assertEqual(ratios, [15.0])
        self.assertEqual(padded_ratio, 15.0)

    def test_transformers_engine_rejects_options_it_cannot_apply(self):
        with self.assertRaisesRegex(ValueError, "only supported by the Paddle"):
            TextRecTransformersPredictor(width_scale=2.0)


if __name__ == "__main__":
    unittest.main()
