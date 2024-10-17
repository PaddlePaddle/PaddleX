# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

from pathlib import Path
from .base import BaseResult, CVResult


class SealOCRResult(CVResult):
    """SealOCRResult"""

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            input_path = self["input_path"]
            save_path = Path(save_path) / f"{Path(input_path).stem}"
        else:
            save_path = Path(save_path).stem
        layout_save_path = f"{save_path}_layout.jpg"
        layout_result = self["layout_result"]
        layout_result.save_to_img(layout_save_path)
        for idx, seal_result in enumerate(self["ocr_result"]):
            ocr_save_path = f"{save_path}_{idx}_seal_ocr.jpg"
            seal_result.save_to_img(ocr_save_path)
