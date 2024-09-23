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

from .base import BasePipeline
from ..predictors import create_predictor
from ...utils import logging
from ..components import CropByPolys
from ..results import OCRResult


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "ocr"

    def __init__(self, det_model, rec_model, det_batch_size, rec_batch_size, **kwargs):
        self._det_predict = create_predictor(det_model, batch_size=det_batch_size)
        self._rec_predict = create_predictor(rec_model, batch_size=rec_batch_size)
        # TODO: foo
        self._crop_by_polys = CropByPolys(det_box_type="foo")

    def predict(self, x):
        batch_ocr_res = []
        for batch_det_res in self._det_predict(x):
            for det_res in batch_det_res:
                single_img_res = det_res["result"]
                single_img_res["rec_text"] = []
                single_img_res["rec_score"] = []
                if len(single_img_res["dt_polys"]) > 0:
                    all_subs_of_img = list(self._crop_by_polys(single_img_res))
                    for batch_rec_res in self._rec_predict(all_subs_of_img):
                        for rec_res in batch_rec_res:
                            single_img_res["rec_text"].append(
                                rec_res["result"]["rec_text"]
                            )
                            single_img_res["rec_score"].append(
                                rec_res["result"]["rec_score"]
                            )
                batch_ocr_res.append({"result": OCRResult(single_img_res)})
        yield batch_ocr_res
