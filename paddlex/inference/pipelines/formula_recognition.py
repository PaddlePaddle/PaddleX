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

import numpy as np
from ..components import CropByBoxes
from ..results import FormulaResult
from .base import BasePipeline
from ...utils import logging


class FormulaRecognitionPipeline(BasePipeline):
    """Formula Recognition Pipeline"""

    entities = "formula_recognition"

    def __init__(
        self,
        layout_model,
        formula_rec_model,
        layout_batch_size=1,
        formula_rec_batch_size=1,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(device, predictor_kwargs)
        self._build_predictor(layout_model, formula_rec_model)
        self.set_predictor(
            layout_batch_size=layout_batch_size,
            formula_rec_batch_size=formula_rec_batch_size,
        )

    def _build_predictor(self, layout_model, formula_rec_model):
        self.layout_predictor = self._create(model=layout_model)
        self.formula_predictor = self._create(model=formula_rec_model)
        self._crop_by_boxes = CropByBoxes()

    def set_predictor(
        self, layout_batch_size=None, formula_rec_batch_size=None, device=None
    ):
        if layout_batch_size:
            self.layout_predictor.set_predictor(batch_size=layout_batch_size)
        if formula_rec_batch_size:
            self.formula_predictor.set_predictor(batch_size=formula_rec_batch_size)
        if device:
            self.layout_predictor.set_predictor(device=device)
            self.formula_predictor.set_predictor(device=device)

    def predict(self, x, **kwargs):
        self.set_predictor(**kwargs)
        for layout_pred in self.layout_predictor(x):
            single_img_res = {
                "input_path": "",
                "layout_result": {},
                "ocr_result": {},
                "table_result": [],
            }
            # update layout result
            single_img_res["input_path"] = layout_pred["input_path"]
            single_img_res["layout_result"] = layout_pred
            single_img_res["dt_polys"] = []
            single_img_res["rec_formula"] = []
            all_subs_of_formula_img = []
            layout_pred["boxes"] = sorted(
                layout_pred["boxes"], key=lambda x: self.sorted_formula_box(x)
            )
            if len(layout_pred["boxes"]) > 0:
                subs_of_img = list(self._crop_by_boxes(layout_pred))
                # get cropped images with label "formula"
                for sub in subs_of_img:
                    if sub["label"].lower() == "formula":
                        boxes = sub["box"]
                        x1, y1, x2, y2 = list(boxes)
                        poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                        all_subs_of_formula_img.append(sub["img"])
                        single_img_res["dt_polys"].append(poly)
                if len(all_subs_of_formula_img) > 0:
                    for formula_res in self.formula_predictor(all_subs_of_formula_img):
                        single_img_res["rec_formula"].append(
                            str(formula_res["rec_text"])
                        )
            yield FormulaResult(single_img_res)

    def sorted_formula_box(self, x):
        coordinate = x["coordinate"]
        x1, y1, x2, y2 = list(coordinate)
        return (y1 + y2) / 2
