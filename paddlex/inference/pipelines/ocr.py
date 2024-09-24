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

from ..components import SortBoxes, CropByPolys
from ..results import OCRResult
from .base import BasePipeline


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "OCR"

    def __init__(
        self,
        det_model,
        rec_model,
        rec_batch_size=1,
        device="gpu",
        predictor_kwargs=None,
    ):
        super().__init__(predictor_kwargs)
        self._det_predict = self._create_predictor(det_model, device=device)
        self._rec_predict = self._create_predictor(
            rec_model, batch_size=rec_batch_size, device=device
        )
        self.is_curve = self._det_predict.model_name in [
            "PP-OCRv4_mobile_seal_det",
            "PP-OCRv4_server_seal_det",
        ]
        self._sort_boxes = SortBoxes()
        self._crop_by_polys = CropByPolys(
            det_box_type="poly" if self.is_curve else "quad"
        )

    def predict(self, x):
        for det_res in self._det_predict(x):
            single_img_res = (
                det_res if self.is_curve else next(self._sort_boxes(det_res))
            )
            single_img_res["rec_text"] = []
            single_img_res["rec_score"] = []
            if len(single_img_res["dt_polys"]) > 0:
                all_subs_of_img = list(self._crop_by_polys(single_img_res))
                for rec_res in self._rec_predict(all_subs_of_img):
                    single_img_res["rec_text"].append(rec_res["rec_text"])
                    single_img_res["rec_score"].append(rec_res["rec_score"])
            yield OCRResult(single_img_res)
