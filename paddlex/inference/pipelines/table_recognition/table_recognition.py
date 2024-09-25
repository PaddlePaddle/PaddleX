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
        batch_size=1,
        device="gpu",
        chat_ocr=False,
        predictor_kwargs=None,
    ):
        super().__init__(predictor_kwargs)

        self.layout_predictor = self._create_model(
            model=layout_model, device=device, batch_size=batch_size
        )

        self.ocr_pipeline = OCRPipeline(
            text_det_model,
            text_rec_model,
            rec_batch_size=batch_size,
            rec_device=device,
            det_device=device,
            predictor_kwargs=predictor_kwargs,
        )
        self.table_predictor = self._create_model(
            model=table_model, device=device, batch_size=batch_size
        )
        self._crop_by_boxes = CropByBoxes()
        self._match = TableMatch(filter_ocr_result=False)
        self.chat_ocr = chat_ocr

    def predict(self, x):
        batch_structure_res = []
        for batch_layout_pred, batch_ocr_pred in zip(
            self.layout_predictor(x), self.ocr_pipeline(x)
        ):
            for layout_pred, ocr_pred in zip(batch_layout_pred, batch_ocr_pred):
                single_img_res = {
                    "img_path": "",
                    "layout_result": {},
                    "ocr_result": {},
                    "table_result": [],
                }
                layout_res = layout_pred["result"]
                # update layout result
                single_img_res["img_path"] = layout_res["img_path"]
                single_img_res["layout_result"] = layout_res
                ocr_res = ocr_pred["result"]
                all_subs_of_img = list(self._crop_by_boxes(layout_res))
                # get cropped images with label 'table'
                table_subs = []
                for batch_subs in all_subs_of_img:
                    table_sub_list = []
                    for sub in batch_subs:
                        box = sub["box"]
                        if sub["label"].lower() == "table":
                            table_sub_list.append(sub)
                            _, ocr_res = self.get_ocr_result_by_bbox(box, ocr_res)
                    table_subs.append(table_sub_list)
                table_res, all_table_ocr_res = self.get_table_result(table_subs)
                for batch_table_ocr_res in all_table_ocr_res:
                    for table_ocr_res in batch_table_ocr_res:
                        ocr_res["dt_polys"].extend(table_ocr_res["dt_polys"])
                        ocr_res["rec_text"].extend(table_ocr_res["rec_text"])
                        ocr_res["rec_score"].extend(table_ocr_res["rec_score"])

                single_img_res["table_result"] = table_res
                single_img_res["ocr_result"] = OCRResult(ocr_res)

                batch_structure_res.append({"result": TableResult(single_img_res)})
        yield batch_structure_res

    def get_ocr_result_by_bbox(self, box, ocr_res):
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

    def get_table_result(self, input_img):
        table_res_list = []
        ocr_res_list = []
        table_index = 0
        for batch_input, batch_table_pred, batch_ocr_pred in zip(
            input_img, self.table_predictor(input_img), self.ocr_pipeline(input_img)
        ):
            batch_table_res = []
            batch_ocr_res = []
            for input, table_pred, ocr_pred in zip(
                batch_input, batch_table_pred, batch_ocr_pred
            ):
                single_table_res = table_pred["result"]
                ocr_res = ocr_pred["result"]
                single_table_box = single_table_res["bbox"]
                ori_x, ori_y, _, _ = input["box"]
                ori_bbox_list = np.array(
                    get_ori_coordinate_for_table(ori_x, ori_y, single_table_box),
                    dtype=np.float32,
                )
                ori_ocr_bbox_list = np.array(
                    get_ori_coordinate_for_table(ori_x, ori_y, ocr_res["dt_polys"]),
                    dtype=np.float32,
                )
                ocr_res["dt_polys"] = ori_ocr_bbox_list
                html_res = self._match(single_table_res, ocr_res)
                batch_table_res.append(
                    StructureTableResult(
                        {
                            "img_path": input["img_path"],
                            "bbox": ori_bbox_list,
                            "img_idx": table_index,
                            "html": html_res,
                        }
                    )
                )
                batch_ocr_res.append(ocr_res)
                table_index += 1
            table_res_list.append(batch_table_res)
            ocr_res_list.append(batch_ocr_res)
        return table_res_list, ocr_res_list
