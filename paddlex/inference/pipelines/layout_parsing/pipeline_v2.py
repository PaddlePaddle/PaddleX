# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...models.object_detection.result import DetResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..ocr.result import OCRResult
from .layout_objects import LayoutBlock, LayoutRegion
from .result_v2 import LayoutParsingResultV2
from .setting import BLOCK_LABEL_MAP, BLOCK_SETTINGS, REGION_SETTINGS
from .utils import (
    caculate_bbox_area,
    calculate_minimum_enclosing_bbox,
    calculate_overlap_ratio,
    convert_formula_res_to_ocr_format,
    gather_imgs,
    get_bbox_intersection,
    get_sub_regions_ocr_res,
    remove_overlap_blocks,
    shrink_supplement_region_bbox,
    update_region_box,
)
from .xycut_enhanced import xycut_enhanced


class _LayoutParsingPipelineV2(BasePipeline):
    """Layout Parsing Pipeline V2"""

    def __init__(
        self,
        config: dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the layout parsing pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
        """

        super().__init__(
            device=device,
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
        )

        self.inintial_predictor(config)

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

    def inintial_predictor(self, config: dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
        self.use_table_recognition = config.get("use_table_recognition", True)
        self.use_seal_recognition = config.get("use_seal_recognition", True)
        self.use_region_detection = config.get(
            "use_region_detection",
            True,
        )
        self.use_formula_recognition = config.get(
            "use_formula_recognition",
            True,
        )
        self.use_chart_recognition = config.get(
            "use_chart_recognition",
            False,
        )

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!",
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config,
            )
        if self.use_region_detection:
            region_detection_config = config.get("SubModules", {}).get(
                "RegionDetection",
                {
                    "model_config_error": "config error for block_region_detection_model!"
                },
            )
            self.region_detection_model = self.create_model(
                region_detection_config,
            )

        layout_det_config = config.get("SubModules", {}).get(
            "LayoutDetection",
            {"model_config_error": "config error for layout_det_model!"},
        )
        layout_kwargs = {}
        if (threshold := layout_det_config.get("threshold", None)) is not None:
            layout_kwargs["threshold"] = threshold
        if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
            layout_kwargs["layout_nms"] = layout_nms
        if (
            layout_unclip_ratio := layout_det_config.get("layout_unclip_ratio", None)
        ) is not None:
            layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
        if (
            layout_merge_bboxes_mode := layout_det_config.get(
                "layout_merge_bboxes_mode", None
            )
        ) is not None:
            layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
        self.layout_det_model = self.create_model(layout_det_config, **layout_kwargs)

        general_ocr_config = config.get("SubPipelines", {}).get(
            "GeneralOCR",
            {"pipeline_config_error": "config error for general_ocr_pipeline!"},
        )
        self.general_ocr_pipeline = self.create_pipeline(
            general_ocr_config,
        )

        if self.use_seal_recognition:
            seal_recognition_config = config.get("SubPipelines", {}).get(
                "SealRecognition",
                {
                    "pipeline_config_error": "config error for seal_recognition_pipeline!",
                },
            )
            self.seal_recognition_pipeline = self.create_pipeline(
                seal_recognition_config,
            )

        if self.use_table_recognition:
            table_recognition_config = config.get("SubPipelines", {}).get(
                "TableRecognition",
                {
                    "pipeline_config_error": "config error for table_recognition_pipeline!",
                },
            )
            self.table_recognition_pipeline = self.create_pipeline(
                table_recognition_config,
            )

        if self.use_formula_recognition:
            formula_recognition_config = config.get("SubPipelines", {}).get(
                "FormulaRecognition",
                {
                    "pipeline_config_error": "config error for formula_recognition_pipeline!",
                },
            )
            self.formula_recognition_pipeline = self.create_pipeline(
                formula_recognition_config,
            )

        if self.use_chart_recognition:
            chart_recognition_config = config.get("SubModules", {}).get(
                "ChartRecognition",
                {
                    "model_config_error": "config error for block_region_detection_model!"
                },
            )
            self.chart_recognition_model = self.create_model(
                chart_recognition_config,
            )

        return

    def get_text_paragraphs_ocr_res(
        self,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
    ) -> OCRResult:
        """
        Retrieves the OCR results for text paragraphs, excluding those of formulas, tables, and seals.

        Args:
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            layout_det_res (DetResult): The detection result containing the layout information of the document.

        Returns:
            OCRResult: The OCR result for text paragraphs after excluding formulas, tables, and seals.
        """
        object_boxes = []
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["formula", "table", "seal"]:
                object_boxes.append(box_info["coordinate"])
        object_boxes = np.array(object_boxes)
        sub_regions_ocr_res = get_sub_regions_ocr_res(
            overall_ocr_res, object_boxes, flag_within=False
        )
        return sub_regions_ocr_res

    def check_model_settings_valid(self, input_params: dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized.",
            )
            return False

        if input_params["use_seal_recognition"] and not self.use_seal_recognition:
            logging.error(
                "Set use_seal_recognition, but the models for seal recognition are not initialized.",
            )
            return False

        if input_params["use_table_recognition"] and not self.use_table_recognition:
            logging.error(
                "Set use_table_recognition, but the models for table recognition are not initialized.",
            )
            return False

        return True

    def standardized_data(
        self,
        image: list,
        region_det_res: DetResult,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        formula_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> list:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.
        Args:
            image (list): The input image.
            overall_ocr_res (OCRResult): An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
                - "input_img": The image on which OCR was performed.
                - "dt_boxes": A list of detected text box coordinates.
                - "rec_texts": A list of recognized text corresponding to the detected boxes.

            layout_det_res (DetResult): An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
                - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "block_label" for the type of content.

            table_res_list (list): A list of table detection results, where each item is a dictionary containing:
                - "block_bbox": The bounding box of the table layout.
                - "pred_html": The predicted HTML representation of the table.

            formula_res_list (list): A list of formula recognition results.
            text_rec_model (Any): The text recognition model.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.
        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """

        matched_ocr_dict = {}
        region_to_block_map = {}
        block_to_ocr_map = {}
        object_boxes = []
        footnote_list = []
        paragraph_title_list = []
        bottom_text_y_max = 0
        max_block_area = 0.0
        doc_title_num = 0

        base_region_bbox = [65535, 65535, 0, 0]
        layout_det_res = remove_overlap_blocks(
            layout_det_res,
            threshold=0.5,
            smaller=True,
        )

        # convert formula_res_list to OCRResult format
        convert_formula_res_to_ocr_format(formula_res_list, overall_ocr_res)

        # match layout boxes and ocr boxes and get some information for layout_order_config
        for box_idx, box_info in enumerate(layout_det_res["boxes"]):
            box = box_info["coordinate"]
            label = box_info["label"].lower()
            object_boxes.append(box)
            _, _, _, y2 = box

            # update the region box and max_block_area according to the layout boxes
            base_region_bbox = update_region_box(box, base_region_bbox)
            max_block_area = max(max_block_area, caculate_bbox_area(box))

            # update_layout_order_config_block_index(layout_order_config, label, box_idx)

            # set the label of footnote to text, when it is above the text boxes
            if label == "footnote":
                footnote_list.append(box_idx)
            elif label == "paragraph_title":
                paragraph_title_list.append(box_idx)
            if label == "text":
                bottom_text_y_max = max(y2, bottom_text_y_max)
            if label == "doc_title":
                doc_title_num += 1

            if label not in ["formula", "table", "seal"]:
                _, matched_idxes = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                block_to_ocr_map[box_idx] = matched_idxes
                for matched_idx in matched_idxes:
                    if matched_ocr_dict.get(matched_idx, None) is None:
                        matched_ocr_dict[matched_idx] = [box_idx]
                    else:
                        matched_ocr_dict[matched_idx].append(box_idx)

        # fix the footnote label
        for footnote_idx in footnote_list:
            if (
                layout_det_res["boxes"][footnote_idx]["coordinate"][3]
                < bottom_text_y_max
            ):
                layout_det_res["boxes"][footnote_idx]["label"] = "text"

        # check if there is only one paragraph title and without doc_title
        only_one_paragraph_title = len(paragraph_title_list) == 1 and doc_title_num == 0
        if only_one_paragraph_title:
            paragraph_title_block_area = caculate_bbox_area(
                layout_det_res["boxes"][paragraph_title_list[0]]["coordinate"]
            )
            title_area_max_block_threshold = BLOCK_SETTINGS.get(
                "title_conversion_area_ratio_threshold", 0.3
            )
            if (
                paragraph_title_block_area
                > max_block_area * title_area_max_block_threshold
            ):
                layout_det_res["boxes"][paragraph_title_list[0]]["label"] = "doc_title"

        # Replace the OCR information of the hurdles.
        for overall_ocr_idx, layout_box_ids in matched_ocr_dict.items():
            if len(layout_box_ids) > 1:
                matched_no = 0
                overall_ocr_box = copy.deepcopy(
                    overall_ocr_res["rec_boxes"][overall_ocr_idx]
                )
                overall_ocr_dt_poly = copy.deepcopy(
                    overall_ocr_res["dt_polys"][overall_ocr_idx]
                )
                for box_idx in layout_box_ids:
                    layout_box = layout_det_res["boxes"][box_idx]["coordinate"]
                    crop_box = get_bbox_intersection(overall_ocr_box, layout_box)
                    for ocr_idx in block_to_ocr_map[box_idx]:
                        ocr_box = overall_ocr_res["rec_boxes"][ocr_idx]
                        iou = calculate_overlap_ratio(ocr_box, crop_box, "small")
                        if iou > 0.8:
                            overall_ocr_res["rec_texts"][ocr_idx] = ""
                    x1, y1, x2, y2 = [int(i) for i in crop_box]
                    crop_img = np.array(image)[y1:y2, x1:x2]
                    crop_img_rec_res = list(text_rec_model([crop_img]))[0]
                    crop_img_dt_poly = get_bbox_intersection(
                        overall_ocr_dt_poly, layout_box, return_format="poly"
                    )
                    crop_img_rec_score = crop_img_rec_res["rec_score"]
                    crop_img_rec_text = crop_img_rec_res["rec_text"]
                    text_rec_score_thresh = (
                        text_rec_score_thresh
                        if text_rec_score_thresh is not None
                        else (self.general_ocr_pipeline.text_rec_score_thresh)
                    )
                    if crop_img_rec_score >= text_rec_score_thresh:
                        matched_no += 1
                        if matched_no == 1:
                            # the first matched ocr be replaced by the first matched layout box
                            overall_ocr_res["dt_polys"][
                                overall_ocr_idx
                            ] = crop_img_dt_poly
                            overall_ocr_res["rec_boxes"][overall_ocr_idx] = crop_box
                            overall_ocr_res["rec_polys"][
                                overall_ocr_idx
                            ] = crop_img_dt_poly
                            overall_ocr_res["rec_scores"][
                                overall_ocr_idx
                            ] = crop_img_rec_score
                            overall_ocr_res["rec_texts"][
                                overall_ocr_idx
                            ] = crop_img_rec_text
                        else:
                            # the other matched ocr be appended to the overall ocr result
                            overall_ocr_res["dt_polys"].append(crop_img_dt_poly)
                            if len(overall_ocr_res["rec_boxes"]) == 0:
                                overall_ocr_res["rec_boxes"] = np.array([crop_box])
                            else:
                                overall_ocr_res["rec_boxes"] = np.vstack(
                                    (overall_ocr_res["rec_boxes"], crop_box)
                                )
                            overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                            overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                            overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                            overall_ocr_res["rec_labels"].append("text")
                            block_to_ocr_map[box_idx].remove(overall_ocr_idx)
                            block_to_ocr_map[box_idx].append(
                                len(overall_ocr_res["rec_texts"]) - 1
                            )

        # use layout bbox to do ocr recognition when there is no matched ocr
        for layout_box_idx, overall_ocr_idxes in block_to_ocr_map.items():
            has_text = False
            for idx in overall_ocr_idxes:
                if overall_ocr_res["rec_texts"][idx] != "":
                    has_text = True
                    break
            if not has_text and layout_det_res["boxes"][layout_box_idx][
                "label"
            ] not in BLOCK_LABEL_MAP.get("vision_labels", []):
                crop_box = layout_det_res["boxes"][layout_box_idx]["coordinate"]
                x1, y1, x2, y2 = [int(i) for i in crop_box]
                crop_img = np.array(image)[y1:y2, x1:x2]
                crop_img_rec_res = next(text_rec_model([crop_img]))
                crop_img_dt_poly = get_bbox_intersection(
                    crop_box, crop_box, return_format="poly"
                )
                crop_img_rec_score = crop_img_rec_res["rec_score"]
                crop_img_rec_text = crop_img_rec_res["rec_text"]
                text_rec_score_thresh = (
                    text_rec_score_thresh
                    if text_rec_score_thresh is not None
                    else (self.general_ocr_pipeline.text_rec_score_thresh)
                )
                if crop_img_rec_score >= text_rec_score_thresh:
                    if len(overall_ocr_res["rec_boxes"]) == 0:
                        overall_ocr_res["rec_boxes"] = np.array([crop_box])
                    else:
                        overall_ocr_res["rec_boxes"] = np.vstack(
                            (overall_ocr_res["rec_boxes"], crop_box)
                        )
                    overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                    overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                    overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                    overall_ocr_res["rec_labels"].append("text")
                    block_to_ocr_map[layout_box_idx].append(
                        len(overall_ocr_res["rec_texts"]) - 1
                    )

        # when there is no layout detection result but there is ocr result, convert ocr detection result to layout detection result
        if len(layout_det_res["boxes"]) == 0 and len(overall_ocr_res["rec_boxes"]) > 0:
            for idx, ocr_rec_box in enumerate(overall_ocr_res["rec_boxes"]):
                base_region_bbox = update_region_box(ocr_rec_box, base_region_bbox)
                layout_det_res["boxes"].append(
                    {
                        "label": "text",
                        "coordinate": ocr_rec_box,
                        "score": overall_ocr_res["rec_scores"][idx],
                    }
                )
                block_to_ocr_map[idx] = [idx]

        mask_labels = (
            BLOCK_LABEL_MAP.get("unordered_labels", [])
            + BLOCK_LABEL_MAP.get("header_labels", [])
            + BLOCK_LABEL_MAP.get("footer_labels", [])
        )
        block_bboxes = [box["coordinate"] for box in layout_det_res["boxes"]]
        region_det_res["boxes"] = sorted(
            region_det_res["boxes"],
            key=lambda item: caculate_bbox_area(item["coordinate"]),
        )
        if len(region_det_res["boxes"]) == 0:
            region_det_res["boxes"] = [
                {
                    "coordinate": base_region_bbox,
                    "label": "SupplementaryRegion",
                    "score": 1,
                }
            ]
            region_to_block_map[0] = range(len(block_bboxes))
        else:
            block_idxes_set = set(range(len(block_bboxes)))
            # match block to region
            for region_idx, region_info in enumerate(region_det_res["boxes"]):
                matched_idxes = []
                region_to_block_map[region_idx] = []
                region_bbox = region_info["coordinate"]
                for block_idx in block_idxes_set:
                    if layout_det_res["boxes"][block_idx]["label"] in mask_labels:
                        continue
                    overlap_ratio = calculate_overlap_ratio(
                        region_bbox, block_bboxes[block_idx], mode="small"
                    )
                    if overlap_ratio > REGION_SETTINGS.get(
                        "match_block_overlap_ratio_threshold", 0.8
                    ):
                        matched_idxes.append(block_idx)
                old_region_bbox_matched_idxes = []
                if len(matched_idxes) > 0:
                    while len(old_region_bbox_matched_idxes) != len(matched_idxes):
                        old_region_bbox_matched_idxes = copy.deepcopy(matched_idxes)
                        matched_idxes = []
                        matched_bboxes = [
                            block_bboxes[idx] for idx in old_region_bbox_matched_idxes
                        ]
                        new_region_bbox = calculate_minimum_enclosing_bbox(
                            matched_bboxes
                        )
                        for block_idx in block_idxes_set:
                            if (
                                layout_det_res["boxes"][block_idx]["label"]
                                in mask_labels
                            ):
                                continue
                            overlap_ratio = calculate_overlap_ratio(
                                new_region_bbox, block_bboxes[block_idx], mode="small"
                            )
                            if overlap_ratio > REGION_SETTINGS.get(
                                "match_block_overlap_ratio_threshold", 0.8
                            ):
                                matched_idxes.append(block_idx)
                    for block_idx in matched_idxes:
                        block_idxes_set.remove(block_idx)
                    region_to_block_map[region_idx] = matched_idxes
                    region_det_res["boxes"][region_idx]["coordinate"] = new_region_bbox
            # Supplement region when there is no matched block
            while len(block_idxes_set) > 0:
                unmatched_bboxes = [block_bboxes[idx] for idx in block_idxes_set]
                if len(unmatched_bboxes) == 0:
                    break
                supplement_region_bbox = calculate_minimum_enclosing_bbox(
                    unmatched_bboxes
                )
                matched_idxes = []
                # check if the new region bbox is overlapped with other region bbox, if have, then shrink the new region bbox
                for region_idx, region_info in enumerate(region_det_res["boxes"]):
                    if len(region_to_block_map[region_idx]) == 0:
                        continue
                    region_bbox = region_info["coordinate"]
                    overlap_ratio = calculate_overlap_ratio(
                        supplement_region_bbox, region_bbox
                    )
                    if overlap_ratio > 0:
                        supplement_region_bbox, matched_idxes = (
                            shrink_supplement_region_bbox(
                                supplement_region_bbox,
                                region_bbox,
                                image.shape[1],
                                image.shape[0],
                                block_idxes_set,
                                block_bboxes,
                            )
                        )

                matched_idxes = [
                    idx
                    for idx in matched_idxes
                    if layout_det_res["boxes"][idx]["label"] not in mask_labels
                ]
                if len(matched_idxes) == 0:
                    matched_idxes = [
                        idx
                        for idx in block_idxes_set
                        if layout_det_res["boxes"][idx]["label"] not in mask_labels
                    ]
                    if len(matched_idxes) == 0:
                        break
                matched_bboxes = [block_bboxes[idx] for idx in matched_idxes]
                supplement_region_bbox = calculate_minimum_enclosing_bbox(
                    matched_bboxes
                )
                region_idx = len(region_det_res["boxes"])
                region_to_block_map[region_idx] = list(matched_idxes)
                for block_idx in matched_idxes:
                    block_idxes_set.remove(block_idx)
                region_det_res["boxes"].append(
                    {
                        "coordinate": supplement_region_bbox,
                        "label": "SupplementaryRegion",
                        "score": 1,
                    }
                )

            mask_idxes = [
                idx
                for idx in range(len(layout_det_res["boxes"]))
                if layout_det_res["boxes"][idx]["label"] in mask_labels
            ]
            for idx in mask_idxes:
                bbox = layout_det_res["boxes"][idx]["coordinate"]
                region_idx = len(region_det_res["boxes"])
                region_to_block_map[region_idx] = [idx]
                region_det_res["boxes"].append(
                    {
                        "coordinate": bbox,
                        "label": "SupplementaryRegion",
                        "score": 1,
                    }
                )

        region_block_ocr_idx_map = dict(
            region_to_block_map=region_to_block_map,
            block_to_ocr_map=block_to_ocr_map,
        )

        return region_block_ocr_idx_map, region_det_res, layout_det_res

    def get_layout_parsing_objects(
        self,
        image: list,
        region_block_ocr_idx_map: dict,
        region_det_res: DetResult,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
        table_res_list: list,
        seal_res_list: list,
        chart_res_list: list,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> list:
        """
        Extract structured information from OCR and layout detection results.

        Args:
            image (list): The input image.
            overall_ocr_res (OCRResult): An object containing the overall OCR results, including detected text boxes and recognized text. The structure is expected to have:
                - "input_img": The image on which OCR was performed.
                - "dt_boxes": A list of detected text box coordinates.
                - "rec_texts": A list of recognized text corresponding to the detected boxes.

            layout_det_res (DetResult): An object containing the layout detection results, including detected layout boxes and their labels. The structure is expected to have:
                - "boxes": A list of dictionaries with keys "coordinate" for box coordinates and "block_label" for the type of content.

            table_res_list (list): A list of table detection results, where each item is a dictionary containing:
                - "block_bbox": The bounding box of the table layout.
                - "pred_html": The predicted HTML representation of the table.

            seal_res_list (List): A list of seal detection results. The details of each item depend on the specific application context.
            text_rec_model (Any): A model for text recognition.
            text_rec_score_thresh (Union[float, None]): The minimum score required for a recognized character to be considered valid. If None, use the default value specified during initialization. Default is None.

        Returns:
            list: A list of structured boxes where each item is a dictionary containing:
                - "block_label": The label of the content (e.g., 'table', 'chart', 'image').
                - The label as a key with either table HTML or image data and text.
                - "block_bbox": The coordinates of the layout box.
        """

        table_index = 0
        seal_index = 0
        chart_index = 0
        layout_parsing_blocks: List[LayoutBlock] = []

        for box_idx, box_info in enumerate(layout_det_res["boxes"]):

            label = box_info["label"]
            block_bbox = box_info["coordinate"]
            rec_res = {"boxes": [], "rec_texts": [], "rec_labels": []}

            block = LayoutBlock(label=label, bbox=block_bbox)

            if label == "table" and len(table_res_list) > 0:
                block.content = table_res_list[table_index]["pred_html"]
                table_index += 1
            elif label == "seal" and len(seal_res_list) > 0:
                block.content = "\n".join(seal_res_list[seal_index]["rec_texts"])
                seal_index += 1
            elif label == "chart" and len(chart_res_list) > 0:
                block.content = chart_res_list[chart_index]
                chart_index += 1
            else:
                if label == "formula":
                    _, ocr_idx_list = get_sub_regions_ocr_res(
                        overall_ocr_res, [block_bbox], return_match_idx=True
                    )
                    region_block_ocr_idx_map["block_to_ocr_map"][box_idx] = ocr_idx_list
                else:
                    ocr_idx_list = region_block_ocr_idx_map["block_to_ocr_map"].get(
                        box_idx, []
                    )
                for box_no in ocr_idx_list:
                    rec_res["boxes"].append(overall_ocr_res["rec_boxes"][box_no])
                    rec_res["rec_texts"].append(
                        overall_ocr_res["rec_texts"][box_no],
                    )
                    rec_res["rec_labels"].append(
                        overall_ocr_res["rec_labels"][box_no],
                    )
                block.update_text_content(
                    image=image,
                    ocr_rec_res=rec_res,
                    text_rec_model=text_rec_model,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

            if (
                label
                in ["seal", "table", "formula", "chart"]
                + BLOCK_LABEL_MAP["image_labels"]
            ):
                x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                img_path = (
                    f"imgs/img_in_{block.label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                )
                img = Image.fromarray(image[y_min:y_max, x_min:x_max, ::-1])
                block.image = {"path": img_path, "img": img}

            layout_parsing_blocks.append(block)

        page_region_bbox = [65535, 65535, 0, 0]
        layout_parsing_regions: List[LayoutRegion] = []
        for region_idx, region_info in enumerate(region_det_res["boxes"]):
            region_bbox = np.array(region_info["coordinate"]).astype("int")
            region_blocks = [
                layout_parsing_blocks[idx]
                for idx in region_block_ocr_idx_map["region_to_block_map"][region_idx]
            ]
            if region_blocks:
                page_region_bbox = update_region_box(region_bbox, page_region_bbox)
                region = LayoutRegion(bbox=region_bbox, blocks=region_blocks)
                layout_parsing_regions.append(region)

        layout_parsing_page = LayoutRegion(
            bbox=np.array(page_region_bbox).astype("int"), blocks=layout_parsing_regions
        )

        return layout_parsing_page

    def sort_layout_parsing_blocks(
        self, layout_parsing_page: LayoutRegion
    ) -> List[LayoutBlock]:
        layout_parsing_regions = xycut_enhanced(layout_parsing_page)
        parsing_res_list = []
        for region in layout_parsing_regions:
            layout_parsing_blocks = xycut_enhanced(region)
            parsing_res_list.extend(layout_parsing_blocks)

        return parsing_res_list

    def get_layout_parsing_res(
        self,
        image: list,
        region_det_res: DetResult,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        table_res_list: list,
        seal_res_list: list,
        chart_res_list: list,
        formula_res_list: list,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> list:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.
        Args:
            image (list): The input image.
            layout_det_res (DetResult): The detection result containing the layout information of the document.
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            table_res_list (list): A list of table recognition results.
            seal_res_list (list): A list of seal recognition results.
            formula_res_list (list): A list of formula recognition results.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.
        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """

        # Standardize data
        region_block_ocr_idx_map, region_det_res, layout_det_res = (
            self.standardized_data(
                image=image,
                region_det_res=region_det_res,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                formula_res_list=formula_res_list,
                text_rec_model=self.general_ocr_pipeline.text_rec_model,
                text_rec_score_thresh=text_rec_score_thresh,
            )
        )

        # Format layout parsing block
        layout_parsing_page = self.get_layout_parsing_objects(
            image=image,
            region_block_ocr_idx_map=region_block_ocr_idx_map,
            region_det_res=region_det_res,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            table_res_list=table_res_list,
            seal_res_list=seal_res_list,
            chart_res_list=chart_res_list,
            text_rec_model=self.general_ocr_pipeline.text_rec_model,
            text_rec_score_thresh=self.general_ocr_pipeline.text_rec_score_thresh,
        )

        parsing_res_list = self.sort_layout_parsing_blocks(layout_parsing_page)

        index = 1
        for block in parsing_res_list:
            if block.label in BLOCK_LABEL_MAP["visualize_index_labels"]:
                block.order_index = index
                index += 1

        return parsing_res_list

    def get_model_settings(
        self,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
        use_seal_recognition: Union[bool, None],
        use_table_recognition: Union[bool, None],
        use_formula_recognition: Union[bool, None],
        use_chart_recognition: Union[bool, None],
        use_region_detection: Union[bool, None],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Union[bool, None]): Enables document orientation classification if True. Defaults to system setting if None.
            use_doc_unwarping (Union[bool, None]): Enables document unwarping if True. Defaults to system setting if None.
            use_seal_recognition (Union[bool, None]): Enables seal recognition if True. Defaults to system setting if None.
            use_table_recognition (Union[bool, None]): Enables table recognition if True. Defaults to system setting if None.
            use_formula_recognition (Union[bool, None]): Enables formula recognition if True. Defaults to system setting if None.

        Returns:
            dict: A dictionary containing the model settings.

        """
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = self.use_doc_preprocessor
        else:
            if use_doc_orientation_classify is True or use_doc_unwarping is True:
                use_doc_preprocessor = True
            else:
                use_doc_preprocessor = False

        if use_seal_recognition is None:
            use_seal_recognition = self.use_seal_recognition

        if use_table_recognition is None:
            use_table_recognition = self.use_table_recognition

        if use_formula_recognition is None:
            use_formula_recognition = self.use_formula_recognition

        if use_region_detection is None:
            use_region_detection = self.use_region_detection

        if use_chart_recognition is None:
            use_chart_recognition = self.use_chart_recognition

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_region_detection=use_region_detection,
        )

    def predict(
        self,
        input: Union[str, list[str], np.ndarray, list[np.ndarray]],
        use_doc_orientation_classify: Union[bool, None] = False,
        use_doc_unwarping: Union[bool, None] = False,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Union[bool, None] = None,
        use_table_recognition: Union[bool, None] = None,
        use_formula_recognition: Union[bool, None] = None,
        use_chart_recognition: Union[bool, None] = False,
        use_region_detection: Union[bool, None] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Union[int, None] = None,
        text_det_limit_type: Union[str, None] = None,
        text_det_thresh: Union[float, None] = None,
        text_det_box_thresh: Union[float, None] = None,
        text_det_unclip_ratio: Union[float, None] = None,
        text_rec_score_thresh: Union[float, None] = None,
        seal_det_limit_side_len: Union[int, None] = None,
        seal_det_limit_type: Union[str, None] = None,
        seal_det_thresh: Union[float, None] = None,
        seal_det_box_thresh: Union[float, None] = None,
        seal_det_unclip_ratio: Union[float, None] = None,
        seal_rec_score_thresh: Union[float, None] = None,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = True,
        **kwargs,
    ) -> LayoutParsingResultV2:
        """
        Predicts the layout parsing result for the given input.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.
            use_formula_recognition (Optional[bool]): Whether to use formula recognition.
            use_region_detection (Optional[bool]): Whether to use region detection.
            layout_threshold (Optional[float]): The threshold value to filter out low-confidence predictions. Default is None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[str], optional): The mode for merging bounding boxes. Defaults to None.
            text_det_limit_side_len (Optional[int]): Maximum side length for text detection.
            text_det_limit_type (Optional[str]): Type of limit to apply for text detection.
            text_det_thresh (Optional[float]): Threshold for text detection.
            text_det_box_thresh (Optional[float]): Threshold for text detection boxes.
            text_det_unclip_ratio (Optional[float]): Ratio for unclipping text detection boxes.
            text_rec_score_thresh (Optional[float]): Score threshold for text recognition.
            seal_det_limit_side_len (Optional[int]): Maximum side length for seal detection.
            seal_det_limit_type (Optional[str]): Type of limit to apply for seal detection.
            seal_det_thresh (Optional[float]): Threshold for seal detection.
            seal_det_box_thresh (Optional[float]): Threshold for seal detection boxes.
            seal_det_unclip_ratio (Optional[float]): Ratio for unclipping seal detection boxes.
            seal_rec_score_thresh (Optional[float]): Score threshold for seal recognition.
            use_wired_table_cells_trans_to_html (bool): Whether to use wired table cells trans to HTML.
            use_wireless_table_cells_trans_to_html (bool): Whether to use wireless table cells trans to HTML.
            use_table_orientation_classify (bool): Whether to use table orientation classification.
            use_ocr_results_with_table_cells (bool): Whether to use OCR results processed by table cells.
            use_e2e_wired_table_rec_model (bool): Whether to use end-to-end wired table recognition model.
            use_e2e_wireless_table_rec_model (bool): Whether to use end-to-end wireless table recognition model.
            **kwargs (Any): Additional settings to extend functionality.

        Returns:
            LayoutParsingResultV2: The predicted layout parsing result.
        """
        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_seal_recognition,
            use_table_recognition,
            use_formula_recognition,
            use_chart_recognition,
            use_region_detection,
        )

        if not self.check_model_settings_valid(model_settings):
            yield {"error": "the input params for model settings are invalid!"}

        for batch_data in self.batch_sampler(input):
            image_arrays = self.img_reader(batch_data.instances)

            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_results = list(
                    self.doc_preprocessor_pipeline(
                        image_arrays,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
            else:
                doc_preprocessor_results = [{"output_img": arr} for arr in image_arrays]

            doc_preprocessor_images = [
                item["output_img"] for item in doc_preprocessor_results
            ]

            layout_det_results = list(
                self.layout_det_model(
                    doc_preprocessor_images,
                    threshold=layout_threshold,
                    layout_nms=layout_nms,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                )
            )
            imgs_in_doc = [
                gather_imgs(img, res["boxes"])
                for img, res in zip(doc_preprocessor_images, layout_det_results)
            ]

            if model_settings["use_region_detection"]:
                region_det_results = list(
                    self.region_detection_model(
                        doc_preprocessor_images,
                        layout_nms=True,
                        layout_merge_bboxes_mode="small",
                    ),
                )
            else:
                region_det_results = [{"boxes": []} for _ in doc_preprocessor_images]

            if model_settings["use_formula_recognition"]:
                formula_res_all = list(
                    self.formula_recognition_pipeline(
                        doc_preprocessor_images,
                        use_layout_detection=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        layout_det_res=layout_det_results,
                    ),
                )
                formula_res_lists = [
                    item["formula_res_list"] for item in formula_res_all
                ]
            else:
                formula_res_lists = [[] for _ in doc_preprocessor_images]

            for doc_preprocessor_image, formula_res_list in zip(
                doc_preprocessor_images, formula_res_lists
            ):
                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = 255.0

            overall_ocr_results = list(
                self.general_ocr_pipeline(
                    doc_preprocessor_images,
                    use_textline_orientation=use_textline_orientation,
                    text_det_limit_side_len=text_det_limit_side_len,
                    text_det_limit_type=text_det_limit_type,
                    text_det_thresh=text_det_thresh,
                    text_det_box_thresh=text_det_box_thresh,
                    text_det_unclip_ratio=text_det_unclip_ratio,
                    text_rec_score_thresh=text_rec_score_thresh,
                ),
            )

            for overall_ocr_res in overall_ocr_results:
                overall_ocr_res["rec_labels"] = ["text"] * len(
                    overall_ocr_res["rec_texts"]
                )

            if model_settings["use_table_recognition"]:
                table_res_lists = []
                for (
                    layout_det_res,
                    doc_preprocessor_image,
                    overall_ocr_res,
                    formula_res_list,
                    imgs_in_doc_for_img,
                ) in zip(
                    layout_det_results,
                    doc_preprocessor_images,
                    overall_ocr_results,
                    formula_res_lists,
                    imgs_in_doc,
                ):
                    table_contents_for_img = copy.deepcopy(overall_ocr_res)
                    for formula_res in formula_res_list:
                        x_min, y_min, x_max, y_max = list(
                            map(int, formula_res["dt_polys"])
                        )
                        poly_points = [
                            (x_min, y_min),
                            (x_max, y_min),
                            (x_max, y_max),
                            (x_min, y_max),
                        ]
                        table_contents_for_img["dt_polys"].append(poly_points)
                        rec_formula = formula_res["rec_formula"]
                        if not rec_formula.startswith("$") or not rec_formula.endswith(
                            "$"
                        ):
                            rec_formula = f"${rec_formula}$"
                        table_contents_for_img["rec_texts"].append(f"{rec_formula}")
                        if table_contents_for_img["rec_boxes"].size == 0:
                            table_contents_for_img["rec_boxes"] = np.array(
                                [formula_res["dt_polys"]]
                            )
                        else:
                            table_contents_for_img["rec_boxes"] = np.vstack(
                                (
                                    table_contents_for_img["rec_boxes"],
                                    [formula_res["dt_polys"]],
                                )
                            )
                        table_contents_for_img["rec_polys"].append(poly_points)
                        table_contents_for_img["rec_scores"].append(1)

                    for img in imgs_in_doc_for_img:
                        img_path = img["path"]
                        x_min, y_min, x_max, y_max = img["coordinate"]
                        poly_points = [
                            (x_min, y_min),
                            (x_max, y_min),
                            (x_max, y_max),
                            (x_min, y_max),
                        ]
                        table_contents_for_img["dt_polys"].append(poly_points)
                        table_contents_for_img["rec_texts"].append(
                            f'<div style="text-align: center;"><img src="{img_path}" alt="Image" /></div>'
                        )
                        if table_contents_for_img["rec_boxes"].size == 0:
                            table_contents_for_img["rec_boxes"] = np.array(
                                [img["coordinate"]]
                            )
                        else:
                            table_contents_for_img["rec_boxes"] = np.vstack(
                                (table_contents_for_img["rec_boxes"], img["coordinate"])
                            )
                        table_contents_for_img["rec_polys"].append(poly_points)
                        table_contents_for_img["rec_scores"].append(img["score"])

                    table_res_all = list(
                        self.table_recognition_pipeline(
                            doc_preprocessor_image,
                            use_doc_orientation_classify=False,
                            use_doc_unwarping=False,
                            use_layout_detection=False,
                            use_ocr_model=False,
                            overall_ocr_res=table_contents_for_img,
                            layout_det_res=layout_det_res,
                            cell_sort_by_y_projection=True,
                            use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
                            use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
                            use_table_orientation_classify=use_table_orientation_classify,
                            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
                            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
                            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
                        ),
                    )
                    single_table_res_lists = [
                        item["table_res_list"] for item in table_res_all
                    ]
                    table_res_lists.extend(single_table_res_lists)
            else:
                table_res_lists = [[] for _ in doc_preprocessor_images]

            if model_settings["use_seal_recognition"]:
                seal_res_all = list(
                    self.seal_recognition_pipeline(
                        doc_preprocessor_images,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        layout_det_res=layout_det_results,
                        seal_det_limit_side_len=seal_det_limit_side_len,
                        seal_det_limit_type=seal_det_limit_type,
                        seal_det_thresh=seal_det_thresh,
                        seal_det_box_thresh=seal_det_box_thresh,
                        seal_det_unclip_ratio=seal_det_unclip_ratio,
                        seal_rec_score_thresh=seal_rec_score_thresh,
                    ),
                )
                seal_res_lists = [item["seal_res_list"] for item in seal_res_all]
            else:
                seal_res_lists = [[] for _ in doc_preprocessor_images]

            for (
                input_path,
                page_index,
                doc_preprocessor_image,
                doc_preprocessor_res,
                layout_det_res,
                region_det_res,
                overall_ocr_res,
                table_res_list,
                seal_res_list,
                formula_res_list,
                imgs_in_doc_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_images,
                doc_preprocessor_results,
                layout_det_results,
                region_det_results,
                overall_ocr_results,
                table_res_lists,
                seal_res_lists,
                formula_res_lists,
                imgs_in_doc,
            ):
                chart_res_list = []
                if model_settings["use_chart_recognition"]:
                    chart_imgs_list = []
                    for bbox in layout_det_res["boxes"]:
                        if bbox["label"] == "chart":
                            x_min, y_min, x_max, y_max = bbox["coordinate"]
                            chart_img = doc_preprocessor_image[
                                int(y_min) : int(y_max), int(x_min) : int(x_max), :
                            ]
                            chart_imgs_list.append({"image": chart_img})

                    for chart_res_batch in self.chart_recognition_model(
                        input=chart_imgs_list
                    ):
                        chart_res_list.append(chart_res_batch["result"])

                parsing_res_list = self.get_layout_parsing_res(
                    doc_preprocessor_image,
                    region_det_res=region_det_res,
                    layout_det_res=layout_det_res,
                    overall_ocr_res=overall_ocr_res,
                    table_res_list=table_res_list,
                    seal_res_list=seal_res_list,
                    chart_res_list=chart_res_list,
                    formula_res_list=formula_res_list,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = formula_res[
                        "input_img"
                    ]

                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "layout_det_res": layout_det_res,
                    "region_det_res": region_det_res,
                    "overall_ocr_res": overall_ocr_res,
                    "table_res_list": table_res_list,
                    "seal_res_list": seal_res_list,
                    "chart_res_list": chart_res_list,
                    "formula_res_list": formula_res_list,
                    "parsing_res_list": parsing_res_list,
                    "imgs_in_doc": imgs_in_doc_for_img,
                    "model_settings": model_settings,
                }
                yield LayoutParsingResultV2(single_img_res)

    def concatenate_markdown_pages(self, markdown_list: list) -> tuple:
        """
        Concatenate Markdown content from multiple pages into a single document.

        Args:
            markdown_list (list): A list containing Markdown data for each page.

        Returns:
            tuple: A tuple containing the processed Markdown text.
        """
        markdown_texts = ""
        previous_page_last_element_paragraph_end_flag = True

        for res in markdown_list:
            # Get the paragraph flags for the current page
            page_first_element_paragraph_start_flag: bool = res[
                "page_continuation_flags"
            ][0]
            page_last_element_paragraph_end_flag: bool = res["page_continuation_flags"][
                1
            ]

            # Determine whether to add a space or a newline
            if (
                not page_first_element_paragraph_start_flag
                and not previous_page_last_element_paragraph_end_flag
            ):
                last_char_of_markdown = markdown_texts[-1] if markdown_texts else ""
                first_char_of_handler = (
                    res["markdown_texts"][0] if res["markdown_texts"] else ""
                )

                # Check if the last character and the first character are Chinese characters
                last_is_chinese_char = (
                    re.match(r"[\u4e00-\u9fff]", last_char_of_markdown)
                    if last_char_of_markdown
                    else False
                )
                first_is_chinese_char = (
                    re.match(r"[\u4e00-\u9fff]", first_char_of_handler)
                    if first_char_of_handler
                    else False
                )
                if not (last_is_chinese_char or first_is_chinese_char):
                    markdown_texts += " " + res["markdown_texts"]
                else:
                    markdown_texts += res["markdown_texts"]
            else:
                markdown_texts += "\n\n" + res["markdown_texts"]
            previous_page_last_element_paragraph_end_flag = (
                page_last_element_paragraph_end_flag
            )

        return markdown_texts


@pipeline_requires_extra("ocr")
class LayoutParsingPipelineV2(AutoParallelImageSimpleInferencePipeline):
    entities = ["PP-StructureV3"]

    @property
    def _pipeline_cls(self):
        return _LayoutParsingPipelineV2

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
