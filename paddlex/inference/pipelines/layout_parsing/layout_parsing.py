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
from ...results import *
from ...components import *
from ..ocr import OCRPipeline
from ....utils import logging
from ..ppchatocrv3.utils import *
from ..table_recognition import _TableRecPipeline
from ..table_recognition.utils import convert_4point2rect, get_ori_coordinate_for_table


class LayoutParsingPipeline(_TableRecPipeline):
    """Layout Analysis Pileline"""

    entities = "layout_parsing"

    def __init__(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        formula_rec_model,
        doc_image_ori_cls_model=None,
        doc_image_unwarp_model=None,
        seal_text_det_model=None,
        layout_batch_size=1,
        text_det_batch_size=1,
        text_rec_batch_size=1,
        table_batch_size=1,
        doc_image_ori_cls_batch_size=1,
        doc_image_unwarp_batch_size=1,
        seal_text_det_batch_size=1,
        formula_rec_batch_size=1,
        recovery=True,
        device=None,
        predictor_kwargs=None,
    ):
        super().__init__(
            device,
            predictor_kwargs,
        )
        self._build_predictor(
            layout_model=layout_model,
            text_det_model=text_det_model,
            text_rec_model=text_rec_model,
            table_model=table_model,
            doc_image_ori_cls_model=doc_image_ori_cls_model,
            doc_image_unwarp_model=doc_image_unwarp_model,
            seal_text_det_model=seal_text_det_model,
            formula_rec_model=formula_rec_model,
        )
        self.set_predictor(
            layout_batch_size=layout_batch_size,
            text_det_batch_size=text_det_batch_size,
            text_rec_batch_size=text_rec_batch_size,
            table_batch_size=table_batch_size,
            doc_image_ori_cls_batch_size=doc_image_ori_cls_batch_size,
            doc_image_unwarp_batch_size=doc_image_unwarp_batch_size,
            seal_text_det_batch_size=seal_text_det_batch_size,
            formula_rec_batch_size=formula_rec_batch_size,
        )
        self.recovery = recovery

    def _build_predictor(
        self,
        layout_model,
        text_det_model,
        text_rec_model,
        table_model,
        formula_rec_model,
        seal_text_det_model=None,
        doc_image_ori_cls_model=None,
        doc_image_unwarp_model=None,
    ):
        super()._build_predictor(
            layout_model, text_det_model, text_rec_model, table_model
        )

        self.formula_predictor = self._create(formula_rec_model)

        if seal_text_det_model:
            self.curve_pipeline = self._create(
                pipeline=OCRPipeline,
                text_det_model=seal_text_det_model,
                text_rec_model=text_rec_model,
            )
        else:
            self.curve_pipeline = None
        if doc_image_ori_cls_model:
            self.oricls_predictor = self._create(doc_image_ori_cls_model)
        else:
            self.oricls_predictor = None
        if doc_image_unwarp_model:
            self.uvdoc_predictor = self._create(doc_image_unwarp_model)
        else:
            self.uvdoc_predictor = None

        self.img_reader = ReadImage(format="RGB")
        self.cropper = CropByBoxes()

    def set_predictor(
        self,
        layout_batch_size=None,
        text_det_batch_size=None,
        text_rec_batch_size=None,
        table_batch_size=None,
        doc_image_ori_cls_batch_size=None,
        doc_image_unwarp_batch_size=None,
        seal_text_det_batch_size=None,
        formula_rec_batch_size=None,
        device=None,
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
        if table_batch_size:
            self.table_predictor.set_predictor(batch_size=table_batch_size)
        if formula_rec_batch_size:
            self.formula_predictor.set_predictor(batch_size=formula_rec_batch_size)
        if self.curve_pipeline and seal_text_det_batch_size:
            self.curve_pipeline.text_det_model.set_predictor(
                batch_size=seal_text_det_batch_size
            )
        if self.oricls_predictor and doc_image_ori_cls_batch_size:
            self.oricls_predictor.set_predictor(batch_size=doc_image_ori_cls_batch_size)
        if self.uvdoc_predictor and doc_image_unwarp_batch_size:
            self.uvdoc_predictor.set_predictor(batch_size=doc_image_unwarp_batch_size)

        if device:
            if self.curve_pipeline:
                self.curve_pipeline.set_predictor(device=device)
            if self.oricls_predictor:
                self.oricls_predictor.set_predictor(device=device)
            if self.uvdoc_predictor:
                self.uvdoc_predictor.set_predictor(device=device)
            self.layout_predictor.set_predictor(device=device)
            self.ocr_pipeline.set_predictor(device=device)

    def predict(
        self,
        inputs,
        use_doc_image_ori_cls_model=True,
        use_doc_image_unwarp_model=True,
        use_seal_text_det_model=True,
        recovery=True,
        **kwargs,
    ):
        self.set_predictor(**kwargs)
        # get oricls and uvdoc results
        img_info_list = list(self.img_reader(inputs))[0]
        oricls_results = []
        if self.oricls_predictor and use_doc_image_ori_cls_model:
            oricls_results = get_oriclas_results(img_info_list, self.oricls_predictor)
        unwarp_result = []
        if self.uvdoc_predictor and use_doc_image_unwarp_model:
            unwarp_result = get_unwarp_results(img_info_list, self.uvdoc_predictor)
        img_list = [img_info["img"] for img_info in img_info_list]
        for idx, (img_info, layout_pred) in enumerate(
            zip(img_info_list, self.layout_predictor(img_list))
        ):
            single_img_res = {
                "input_path": "",
                "layout_result": DetResult({}),
                "ocr_result": OCRResult({}),
                "table_ocr_result": [],
                "table_result": StructureTableResult([]),
                "layout_parsing_result": {},
                "oricls_result": TopkResult({}),
                "formula_result": TextRecResult({}),
                "unwarp_result": DocTrResult({}),
                "curve_result": [],
            }
            # update oricls and uvdoc result
            if oricls_results:
                single_img_res["oricls_result"] = oricls_results[idx]
            if unwarp_result:
                single_img_res["unwarp_result"] = unwarp_result[idx]
            # update layout result
            single_img_res["input_path"] = layout_pred["input_path"]
            single_img_res["layout_result"] = layout_pred
            single_img = img_info["img"]
            table_subs = []
            curve_subs = []
            formula_subs = []
            structure_res = []
            ocr_res_with_layout = []
            if len(layout_pred["boxes"]) > 0:
                subs_of_img = list(self._crop_by_boxes(layout_pred))
                # get cropped images
                for sub in subs_of_img:
                    box = sub["box"]
                    xmin, ymin, xmax, ymax = [int(i) for i in box]
                    mask_flag = True
                    if sub["label"].lower() == "table":
                        table_subs.append(sub)
                    elif sub["label"].lower() == "seal":
                        curve_subs.append(sub)
                    elif sub["label"].lower() == "formula":
                        formula_subs.append(sub)
                    else:
                        if self.recovery and recovery:
                            # TODO: Why use the entire image?
                            wht_im = (
                                np.ones(single_img.shape, dtype=single_img.dtype) * 255
                            )
                            wht_im[ymin:ymax, xmin:xmax, :] = sub["img"]
                            sub_ocr_res = get_ocr_res(self.ocr_pipeline, wht_im)
                        else:
                            sub_ocr_res = get_ocr_res(self.ocr_pipeline, sub)
                            sub_ocr_res["dt_polys"] = get_ori_coordinate_for_table(
                                xmin, ymin, sub_ocr_res["dt_polys"]
                            )
                        layout_label = sub["label"].lower()
                        # Adapt the user label definition to specify behavior.
                        if sub_ocr_res and sub["label"].lower() in [
                            "image",
                            "figure",
                            "img",
                            "fig",
                        ]:
                            get_text_in_image = kwargs.get("get_text_in_image", False)
                            mask_flag = not get_text_in_image
                            text_in_image = ""
                            if get_text_in_image:
                                text_in_image = "".join(sub_ocr_res["rec_text"])
                                ocr_res_with_layout.append(sub_ocr_res)
                            structure_res.append(
                                {
                                    "input_path": sub_ocr_res["input_path"],
                                    "layout_bbox": box,
                                    f"{layout_label}": {
                                        "img": sub["img"],
                                        f"{layout_label}_text": text_in_image,
                                    },
                                }
                            )
                        else:
                            ocr_res_with_layout.append(sub_ocr_res)
                            structure_res.append(
                                {
                                    "input_path": sub_ocr_res["input_path"],
                                    "layout_bbox": box,
                                    f"{layout_label}": "\n".join(
                                        sub_ocr_res["rec_text"]
                                    ),
                                }
                            )
                    if mask_flag:
                        single_img[ymin:ymax, xmin:xmax, :] = 255

            curve_pipeline = self.ocr_pipeline
            if self.curve_pipeline and use_seal_text_det_model:
                curve_pipeline = self.curve_pipeline

            all_curve_res = get_ocr_res(curve_pipeline, curve_subs)
            single_img_res["curve_result"] = all_curve_res
            if isinstance(all_curve_res, dict):
                all_curve_res = [all_curve_res]
            for sub, curve_res in zip(curve_subs, all_curve_res):
                structure_res.append(
                    {
                        "input_path": curve_res["input_path"],
                        "layout_bbox": sub["box"],
                        "seal": "".join(curve_res["rec_text"]),
                    }
                )

            all_formula_res = get_formula_res(self.formula_predictor, formula_subs)
            single_img_res["formula_result"] = all_formula_res
            for sub, formula_res in zip(formula_subs, all_formula_res):
                structure_res.append(
                    {
                        "input_path": formula_res["input_path"],
                        "layout_bbox": sub["box"],
                        "formula": "".join(formula_res["rec_text"]),
                    }
                )

            use_ocr_without_layout = kwargs.get("use_ocr_without_layout", True)
            ocr_res = {
                "dt_polys": [],
                "rec_text": [],
                "input_path": layout_pred["input_path"],
            }

            if use_ocr_without_layout:
                ocr_res = get_ocr_res(self.ocr_pipeline, single_img)
                ocr_res["input_path"] = layout_pred["input_path"]
                for idx, single_dt_poly in enumerate(ocr_res["dt_polys"]):
                    structure_res.append(
                        {
                            "input_path": ocr_res["input_path"],
                            "layout_bbox": convert_4point2rect(single_dt_poly),
                            "text_without_layout": ocr_res["rec_text"][idx],
                        }
                    )
            # update ocr result
            for layout_ocr_res in ocr_res_with_layout:
                ocr_res["dt_polys"].extend(layout_ocr_res["dt_polys"])
                ocr_res["rec_text"].extend(layout_ocr_res["rec_text"])
                ocr_res["rec_score"].extend(layout_ocr_res["rec_score"])
                ocr_res["input_path"] = single_img_res["input_path"]

            all_table_ocr_res = []
            all_table_res, _ = self.get_table_result(table_subs)
            # get table text from html
            structure_res_table, all_table_ocr_res = get_table_text_from_html(
                all_table_res
            )
            structure_res.extend(structure_res_table)

            # sort the layout result by the left top point of the box
            structure_res = sorted_layout_boxes(structure_res, w=single_img.shape[1])
            structure_res = LayoutParsingResult(
                {
                    "input_path": layout_pred["input_path"],
                    "parsing_result": structure_res,
                }
            )

            single_img_res["table_result"] = all_table_res
            single_img_res["ocr_result"] = ocr_res
            single_img_res["table_ocr_result"] = all_table_ocr_res
            single_img_res["layout_parsing_result"] = structure_res

            yield VisualResult(single_img_res)


def get_formula_res(predictor, input):
    """get formula res"""
    res_list = []
    if isinstance(input, list):
        img = [im["img"] for im in input]
    elif isinstance(input, dict):
        img = input["img"]
    else:
        img = input
    for res in predictor(img):
        res_list.append(res)
    return res_list
