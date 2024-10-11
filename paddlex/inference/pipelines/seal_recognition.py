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
from .base import BasePipeline
from .ocr import OCRPipeline
from ..components import CropByBoxes
from ..results import SealOCRResult
from ...utils import logging


def get_ocr_res(pipeline, input):
    """get ocr res"""
    ocr_res_list = []
    if isinstance(input, list):
        img = [im["img"] for im in input]
    elif isinstance(input, dict):
        img = input["img"]
    else:
        img = input
    for ocr_res in pipeline(img):
        ocr_res_list.append(ocr_res)
    return ocr_res_list


class SealOCRPipeline(BasePipeline):
    """Seal Recognition Pipeline"""

    entities = "seal_recognition"

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        layout_batch_size=1,
        text_det_batch_size=1,
        text_rec_batch_size=1,
        predictor_kwargs=None,
    ):
        self.layout_model = layout_model
        self.text_det_model = text_det_model
        self.text_rec_model = text_rec_model
        self.layout_batch_size = layout_batch_size
        self.text_det_batch_size = text_det_batch_size
        self.text_rec_batch_size = text_rec_batch_size
        self.predictor_kwargs = predictor_kwargs
        super().__init__(predictor_kwargs=predictor_kwargs)
        self._build_predictor()

    def _build_predictor(
        self,
    ):
        self.layout_predictor = self._create_model(model=self.layout_model)
        self.ocr_pipeline = OCRPipeline(
            text_det_model=self.text_det_model,
            text_rec_model=self.text_rec_model,
            text_det_batch_size=self.text_det_batch_size,
            text_rec_batch_size=self.text_rec_batch_size,
            predictor_kwargs=self.predictor_kwargs,
        )
        self._crop_by_boxes = CropByBoxes()
        self.layout_predictor.set_predictor(batch_size=self.layout_batch_size)
        self.ocr_pipeline.text_rec_model.set_predictor(
            batch_size=self.text_rec_batch_size
        )

    def set_predictor(
        self,
        layout_batch_size=None,
        text_det_batch_size=None,
        text_rec_batch_size=None,
        # device=None,
    ):
        if text_det_batch_size and text_det_batch_size > 1:
            logging.warning(
                f"text det model only support batch_size=1 now,the setting of text_det_batch_size={text_det_batch_size} will not using! "
            )
        if layout_batch_size:
            self.layout_predictor.set_predictor(batch_size=layout_batch_size)
        if text_rec_batch_size:
            self.ocr_pipeline.text_rec_model.set_predictor(
                batch_size=text_rec_batch_size
            )

    def predict(self, x, **kwargs):
        layout_batch_size = kwargs.get("layout_batch_size")
        text_det_batch_size = kwargs.get("text_det_batch_size")
        text_rec_batch_size = kwargs.get("text_rec_batch_size")
        # device = kwargs.get("device")
        self.set_predictor(
            layout_batch_size,
            text_det_batch_size,
            text_rec_batch_size,
            # device,
        )
        for layout_pred in self.layout_predictor(x):
            single_img_res = {
                "input_path": "",
                "layout_result": {},
                "ocr_result": {},
            }
            # update layout result
            single_img_res["input_path"] = layout_pred["input_path"]
            single_img_res["layout_result"] = layout_pred

            seal_subs = []
            if len(layout_pred["boxes"]) > 0:
                subs_of_img = list(self._crop_by_boxes(layout_pred))
                # get cropped images with label "seal"
                for sub in subs_of_img:
                    box = sub["box"]
                    if sub["label"].lower() == "seal":
                        seal_subs.append(sub)
            all_seal_ocr_res = get_ocr_res(self.ocr_pipeline, seal_subs)
            single_img_res["ocr_result"] = all_seal_ocr_res
            yield SealOCRResult(single_img_res)
