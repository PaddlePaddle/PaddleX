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
from ...utils import logging


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "OCR"

    def __init__(
        self,
        text_det_model,
        text_rec_model,
        text_det_batch_size=1,
        text_rec_batch_size=1,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(device, predictor_kwargs)
        self._build_predictor(text_det_model, text_rec_model)
        self.set_predictor(
            text_det_batch_size=text_det_batch_size,
            text_rec_batch_size=text_rec_batch_size,
        )

    def _build_predictor(self, text_det_model, text_rec_model):
        self.text_det_model = self._create(model=text_det_model)
        self.text_rec_model = self._create(model=text_rec_model)
        self.is_curve = self.text_det_model.model_name in [
            "PP-OCRv4_mobile_seal_det",
            "PP-OCRv4_server_seal_det",
        ]
        self._sort_boxes = SortBoxes()
        self._crop_by_polys = CropByPolys(
            det_box_type="poly" if self.is_curve else "quad"
        )

    def set_predictor(
        self, text_det_batch_size=None, text_rec_batch_size=None, device=None
    ):
        if text_det_batch_size and text_det_batch_size > 1:
            logging.warning(
                f"text det model only support batch_size=1 now,the setting of text_det_batch_size={text_det_batch_size} will not using! "
            )
        if text_rec_batch_size:
            self.text_rec_model.set_predictor(batch_size=text_rec_batch_size)
        if device:
            self.text_rec_model.set_predictor(device=device)
            self.text_det_model.set_predictor(device=device)

    def predict(self, input, **kwargs):
        self.set_predictor(**kwargs)
        for det_res in self.text_det_model(input):
            single_img_res = (
                det_res if self.is_curve else next(self._sort_boxes(det_res))
            )
            single_img_res["rec_text"] = []
            single_img_res["rec_score"] = []
            if len(single_img_res["dt_polys"]) > 0:
                all_subs_of_img = [
                    sub["img"] for sub in self._crop_by_polys(single_img_res)
                ]
                for rec_res in self.text_rec_model(all_subs_of_img):
                    single_img_res["rec_text"].append(rec_res["rec_text"])
                    single_img_res["rec_score"].append(rec_res["rec_score"])
            yield OCRResult(single_img_res)
