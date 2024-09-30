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
from .base import BaseResult
from .utils.mixin import Base64Mixin


class LayoutStructureResult(BaseResult):
    """LayoutStructureResult"""

    pass


class VisualInfoResult(BaseResult):
    """VisualInfoResult"""

    pass


class VisualResult(BaseResult):
    """VisualInfoResult"""

    def save_to_html(self, save_path):
        if not save_path.lower().endswith(("html")):
            input_path = self["input_path"]
            save_path = Path(save_path) / f"{Path(input_path).stem}"
        else:
            save_path = Path(save_path).stem
        for table_result in self["table_result"]:
            table_result.save_to_html(save_path)

    def save_to_xlsx(self, save_path):
        if not save_path.lower().endswith(("xlsx")):
            input_path = self["input_path"]
            save_path = Path(save_path) / f"{Path(input_path).stem}"
        else:
            save_path = Path(save_path).stem
        for table_result in self["table_result"]:
            table_result.save_to_xlsx(save_path)

    def save_to_img(self, save_path):
        if not save_path.lower().endswith((".jpg", ".png")):
            input_path = self["input_path"]
            save_path = Path(save_path) / f"{Path(input_path).stem}"
        else:
            save_path = Path(save_path).stem

        oricls_save_path = f"{save_path}_oricls.jpg"
        oricls_result = self["oricls_result"]
        oricls_result.save_to_img(oricls_save_path)
        uvdoc_save_path = f"{save_path}_uvdoc.jpg"
        uvdoc_result = self["uvdoc_result"]
        uvdoc_result.save_to_img(uvdoc_save_path)
        curve_save_path = f"{save_path}_curve.jpg"
        for curve_result in self["curve_result"]:
            curve_result.save_to_img(curve_save_path)
        layout_save_path = f"{save_path}_layout.jpg"
        layout_result = self["layout_result"]
        layout_result.save_to_img(layout_save_path)
        ocr_save_path = f"{save_path}_ocr.jpg"
        table_save_path = f"{save_path}_table.jpg"
        ocr_result = self["ocr_result"]
        ocr_result.save_to_img(ocr_save_path)
        for table_result in self["table_result"]:
            table_result.save_to_img(table_save_path)


class VectorResult(BaseResult, Base64Mixin):
    """VisualInfoResult"""

    def _to_base64(self):
        return self["vector"]


class RetrievalResult(BaseResult):
    """VisualInfoResult"""

    pass


class ChatResult(BaseResult):
    """VisualInfoResult"""

    pass
