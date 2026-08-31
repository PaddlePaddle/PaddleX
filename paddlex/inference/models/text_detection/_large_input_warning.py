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


LARGE_DET_INPUT_WARN_PIXELS = 4_000_000


class LargeDetectorInputWarner:
    def __init__(self):
        self._emitted = False

    def warn(self, source_shape, detector_shape, warning):
        detector_h, detector_w = detector_shape[:2]
        if self._emitted or detector_h * detector_w <= LARGE_DET_INPUT_WARN_PIXELS:
            return

        source_h, source_w = source_shape[:2]
        self._emitted = True
        warning(
            f"Text detection preprocessing produced a large detector input of "
            f"{detector_w}x{detector_h} "
            f"({detector_w * detector_h / 1e6:.1f} megapixels) from a "
            f"{source_w}x{source_h} source image. This may increase memory use "
            f"or cause an out-of-memory error. Consider lowering the detector "
            f"input-size limits in the pipeline or model configuration."
        )
