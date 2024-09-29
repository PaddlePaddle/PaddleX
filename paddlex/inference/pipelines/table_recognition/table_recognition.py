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
from ..base import BasePipeline
from ..ocr import OCRPipeline
from ...components import CropByBoxes
from ...results import OCRResult, TableResult, StructureTableResult
from .utils import *


class TableRecPipeline(BasePipeline):
    """Table Recognition Pipeline"""

    entities = "table_recognition"

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        layout_batch_size=1,
        text_rec_batch_size=1,
        table_batch_size=1,
        predictor_kwargs=None,
    ):
        super().__init__(predictor_kwargs=predictor_kwargs)
        self._build_predictor(
            layout_model, text_det_model, text_rec_model, table_model, predictor_kwargs
        )
        self.set_predictor(layout_batch_size, text_rec_batch_size, table_batch_size)

    def _build_predictor(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        predictor_kwargs,
    ):
        self.layout_predictor = self._create_model(model=layout_model)
        self.ocr_pipeline = OCRPipeline(
            text_det_model,
            text_rec_model,
            predictor_kwargs=predictor_kwargs,
        )
        self.table_predictor = self._create_model(model=table_model)
        self._crop_by_boxes = CropByBoxes()
        self._match = TableMatch(filter_ocr_result=False)

    def set_predictor(self, layout_batch_size, text_rec_batch_size, table_batch_size):
        self.layout_predictor.set_predictor(batch_size=layout_batch_size)
        self.ocr_pipeline.rec_model.set_predictor(batch_size=text_rec_batch_size)
        self.table_predictor.set_predictor(batch_size=table_batch_size)

    def predict(self, x):
        for layout_pred, ocr_pred in zip(
            self.layout_predictor(x), self.ocr_pipeline(x)
        ):
            single_img_res = {
                "img_path": "",
                "layout_result": {},
                "ocr_result": {},
                "table_result": [],
            }
            # update layout result
            single_img_res["img_path"] = layout_pred["img_path"]
            single_img_res["layout_result"] = layout_pred
            subs_of_img = list(self._crop_by_boxes(layout_pred))
            # get cropped images with label "table"
            table_subs = []
            for sub in subs_of_img:
                box = sub["box"]
                if sub["label"].lower() == "table":
                    table_subs.append(sub)
                    _, ocr_res = self.get_related_ocr_result(box, ocr_pred)
            table_res, all_table_ocr_res = self.get_table_result(table_subs)
            for table_ocr_res in all_table_ocr_res:
                ocr_res["dt_polys"].extend(table_ocr_res["dt_polys"])
                ocr_res["rec_text"].extend(table_ocr_res["rec_text"])
                ocr_res["rec_score"].extend(table_ocr_res["rec_score"])

            single_img_res["table_result"] = table_res
            single_img_res["ocr_result"] = OCRResult(ocr_res)

            yield TableResult(single_img_res)

    def get_related_ocr_result(self, box, ocr_res):
        dt_polys_list = []
        rec_text_list = []
        score_list = []
        unmatched_ocr_res = {"dt_polys": [], "rec_text": [], "rec_score": []}
        unmatched_ocr_res["img_path"] = ocr_res["img_path"]
        for i, text_box in enumerate(ocr_res["dt_polys"]):
            text_box_area = convert_4point2rect(text_box)
            if is_inside(text_box_area, box):
                dt_polys_list.append(text_box)
                rec_text_list.append(ocr_res["rec_text"][i])
                score_list.append(ocr_res["rec_score"][i])
            else:
                unmatched_ocr_res["dt_polys"].append(text_box)
                unmatched_ocr_res["rec_text"].append(ocr_res["rec_text"][i])
                unmatched_ocr_res["rec_score"].append(ocr_res["rec_score"][i])
        return (dt_polys_list, rec_text_list, score_list), unmatched_ocr_res

    def get_table_result(self, input_imgs):
        table_res_list = []
        ocr_res_list = []
        table_index = 0
        img_list = [img["img"] for img in input_imgs]
        for input_img, table_pred, ocr_pred in zip(
            input_imgs, self.table_predictor(img_list), self.ocr_pipeline(img_list)
        ):
            single_table_box = table_pred["bbox"]
            ori_x, ori_y, _, _ = input_img["box"]
            ori_bbox_list = np.array(
                get_ori_coordinate_for_table(ori_x, ori_y, single_table_box),
                dtype=np.float32,
            )
            ori_ocr_bbox_list = np.array(
                get_ori_coordinate_for_table(ori_x, ori_y, ocr_pred["dt_polys"]),
                dtype=np.float32,
            )
            html_res = self._match(table_pred, ocr_pred)
            ocr_pred["dt_polys"] = ori_ocr_bbox_list
            table_res_list.append(
                StructureTableResult(
                    {
                        "img_path": input_img["img_path"],
                        "layout_bbox": [int(x) for x in input_img["box"]],
                        "bbox": ori_bbox_list,
                        "img_idx": table_index,
                        "html": html_res,
                    }
                )
            )
            ocr_res_list.append(ocr_pred)
            table_index += 1
        return table_res_list, ocr_res_list
