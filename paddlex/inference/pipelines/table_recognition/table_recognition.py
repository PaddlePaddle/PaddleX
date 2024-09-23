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

import re
import numpy as np
from ..base import BasePipeline
from ...predictors import create_predictor
from ..ocr import OCRPipeline
from ...components import CropByBoxes
from ...results import OCRResult, TableResult, StructureTableResult
from copy import deepcopy
from .utils import *


class TableRecPipeline(BasePipeline):
    """Table Recognition Pipeline"""

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        batch_size=1,
        device="gpu",
        chat_ocr=False,
    ):

        self.layout_predictor = create_predictor(
            model=layout_model, device=device, batch_size=batch_size
        )
        self.ocr_pipeline = OCRPipeline(
            text_det_model, text_rec_model, batch_size, device
        )
        self.table_predictor = create_predictor(
            model=table_model, device=device, batch_size=batch_size
        )
        self._crop_by_boxes = CropByBoxes()
        self._match = TableMatch(filter_ocr_result=False)
        self.chat_ocr = chat_ocr
        super().__init__()

    def predict(self, x):
        batch_structure_res = []
        for batch_layout_pred, batch_ocr_pred in zip(
            self.layout_predictor(x), self.ocr_pipeline(x)
        ):
            for layout_pred, ocr_pred in zip(batch_layout_pred, batch_ocr_pred):
                single_img_structure_res = {
                    "img_path": "",
                    "layout_result": {},
                    "ocr_result": {},
                    "table_result": [],
                }
                layout_res = layout_pred["result"]
                # update layout result
                single_img_structure_res["img_path"] = layout_res["img_path"]
                single_img_structure_res["layout_result"] = layout_res
                single_img_ocr_res = ocr_pred["result"]
                all_subs_of_img = list(self._crop_by_boxes(layout_res))
                table_subs_of_img = []
                seal_subs_of_img = []
                # ocr result without table and seal
                ocr_res = deepcopy(single_img_ocr_res)
                # ocr result in table and seal, is for batch
                table_ocr_res, seal_ocr_res = [], []
                # get cropped images and ocr result
                for batch_subs in all_subs_of_img:
                    table_batch_list, seal_batch_list = [], []
                    table_batch_ocr_res, seal_batch_ocr_res = [], []
                    for sub in batch_subs:
                        box = sub["box"]
                        if sub["label"].lower() == "table":
                            table_batch_list.append(sub)
                            relative_res, ocr_res = self.get_ocr_result_by_bbox(
                                box, ocr_res
                            )
                            table_batch_ocr_res.append(
                                {
                                    "dt_polys": relative_res[0],
                                    "rec_text": relative_res[1],
                                }
                            )
                        elif sub["label"].lower() == "seal":
                            seal_batch_list.append(sub)
                            relative_res, ocr_res = self.get_ocr_result_by_bbox(
                                box, ocr_res
                            )
                            seal_batch_ocr_res.append(
                                {
                                    "dt_polys": relative_res[0],
                                    "rec_text": relative_res[1],
                                }
                            )
                        elif sub["label"].lower() == "figure":
                            # remove ocr result in figure
                            _, ocr_res = self.get_ocr_result_by_bbox(box, ocr_res)
                    table_subs_of_img.append(table_batch_list)
                    table_ocr_res.append(table_batch_ocr_res)
                    seal_subs_of_img.append(seal_batch_list)
                    seal_ocr_res.append(seal_batch_ocr_res)

                # get table result
                table_res = self.get_table_result(table_subs_of_img, table_ocr_res)
                # get seal result
                if seal_subs_of_img:
                    pass

                if self.chat_ocr:
                    # chat ocr does not visualize table results in ocr result
                    single_img_structure_res["ocr_result"] = OCRResult(ocr_res)
                else:
                    single_img_structure_res["ocr_result"] = single_img_ocr_res
                single_img_structure_res["table_result"] = table_res
                batch_structure_res.append(
                    {"result": TableResult(single_img_structure_res)}
                )
        yield batch_structure_res

    def get_ocr_result_by_bbox(self, box, ocr_res):
        dt_polys_list = []
        rec_text_list = []
        unmatched_ocr_res = {"dt_polys": [], "rec_text": []}
        for text_box, text_res in zip(ocr_res["dt_polys"], ocr_res["rec_text"]):
            text_box_area = convert_4point2rect(text_box)
            if is_inside(box, text_box_area):
                dt_polys_list.append(text_box)
                rec_text_list.append(text_res)
            else:
                unmatched_ocr_res["dt_polys"].append(text_box)
                unmatched_ocr_res["rec_text"].append(text_res)
        return (dt_polys_list, rec_text_list), unmatched_ocr_res

    def get_table_result(self, input_img, table_ocr_res):
        table_res_list = []
        table_index = 0
        for batch_input, batch_table_res, batch_ocr_res in zip(
            input_img, self.table_predictor(input_img), table_ocr_res
        ):
            batch_res_list = []
            for roi_img, table_res, ocr_res in zip(
                batch_input, batch_table_res, batch_ocr_res
            ):
                single_table_res = table_res["result"]
                single_table_box = single_table_res["bbox"]
                ori_x, ori_y, _, _ = roi_img["box"]
                ori_bbox_list = np.array(
                    get_ori_coordinate_for_table(ori_x, ori_y, single_table_box),
                    dtype=np.float32,
                )
                single_table_res["bbox"] = ori_bbox_list
                html_res = self._match(single_table_res, ocr_res)
                batch_res_list.append(
                    StructureTableResult(
                        {
                            "img_path": roi_img["img_path"],
                            "img_idx": table_index,
                            "bbox": ori_bbox_list,
                            "html": html_res,
                            "structure": single_table_res["structure"],
                        }
                    )
                )
                table_index += 1
            table_res_list.append(batch_res_list)
        return table_res_list
