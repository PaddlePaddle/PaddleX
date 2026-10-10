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

import importlib.util
from pathlib import Path
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "large_input_warning",
    REPO_ROOT / "paddlex/inference/models/text_detection/_large_input_warning.py",
)
large_input_warning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(large_input_warning)


LARGE_DET_INPUT_WARN_PIXELS = large_input_warning.LARGE_DET_INPUT_WARN_PIXELS
LargeDetectorInputWarner = large_input_warning.LargeDetectorInputWarner


def test_large_detector_input_warning_boundary():
    warner = LargeDetectorInputWarner()
    warning = Mock()

    warner.warn(
        (1000, 4000, 3),
        (1000, LARGE_DET_INPUT_WARN_PIXELS // 1000, 3),
        warning,
    )

    warning.assert_not_called()


def test_large_detector_input_warning_is_bounded_per_processor():
    warner = LargeDetectorInputWarner()
    warning = Mock()

    warner.warn((3000, 2000, 3), (3000, 2000, 3), warning)
    warner.warn((4000, 3000, 3), (4000, 3000, 3), warning)

    warning.assert_called_once()


def test_large_detector_input_warning_describes_detector_and_source_paths():
    warner = LargeDetectorInputWarner()
    warning = Mock()

    warner.warn((3000, 4000, 3), (2048, 2048, 3), warning)

    message = warning.call_args.args[0]
    assert "2048x2048" in message
    assert "4000x3000 source image" in message
    assert "without downscaling" not in message
    assert "text_det_limit" not in message
