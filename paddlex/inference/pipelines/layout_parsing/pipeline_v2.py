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
from ..base import BasePipeline
from ..ocr.result import OCRResult
from .result_v2 import LayoutParsingBlock, LayoutParsingResultV2
from .utils import (
    caculate_bbox_area,
    calculate_text_orientation,
    convert_formula_res_to_ocr_format,
    format_line,
    gather_imgs,
    get_bbox_intersection,
    get_sub_regions_ocr_res,
    group_boxes_into_lines,
    remove_overlap_blocks,
    split_boxes_if_x_contained,
    update_layout_order_config_block_index,
    update_region_box,
)
from .xycut_enhanced import xycut_enhanced


@pipeline_requires_extra("ocr")
class LayoutParsingPipelineV2(BasePipeline):
    """Layout Parsing Pipeline V2"""

    entities = ["PP-StructureV3"]

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
                inference plugin (HPIP). Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The high-performance inference configuration dictionary.
                Defaults to None.
        """

        super().__init__(
            device=device,
            pp_option=pp_option,
            use_hpip=use_hpip,
            hpi_config=hpi_config,
        )

        self.inintial_predictor(config)
        self.batch_sampler = ImageBatchSampler(batch_size=1)

        self.img_reader = ReadImage(format="BGR")

    def inintial_predictor(self, config: dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
        self.use_general_ocr = config.get("use_general_ocr", True)
        self.use_table_recognition = config.get("use_table_recognition", True)
        self.use_seal_recognition = config.get("use_seal_recognition", True)
        self.use_formula_recognition = config.get(
            "use_formula_recognition",
            True,
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

        if self.use_general_ocr or self.use_table_recognition:
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

        if input_params["use_general_ocr"] and not self.use_general_ocr:
            logging.error(
                "Set use_general_ocr, but the models for general OCR are not initialized.",
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
        layout_order_config: dict,
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
        layout_to_ocr_mapping = {}
        object_boxes = []
        footnote_list = []
        bottom_text_y_max = 0
        max_block_area = 0.0

        region_box = [65535, 65535, 0, 0]
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
            region_box = update_region_box(box, region_box)
            max_block_area = max(max_block_area, caculate_bbox_area(box))

            update_layout_order_config_block_index(layout_order_config, label, box_idx)

            # set the label of footnote to text, when it is above the text boxes
            if label == "footnote":
                footnote_list.append(box_idx)
            if label == "text":
                bottom_text_y_max = max(y2, bottom_text_y_max)

            if label not in ["formula", "table", "seal"]:
                _, matched_idxes = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                layout_to_ocr_mapping[box_idx] = matched_idxes
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
                layout_order_config["text_block_idxes"].append(footnote_idx)
                layout_order_config["footer_block_idxes"].remove(footnote_idx)

        # fix the doc_title label
        doc_title_idxes = layout_order_config.get("doc_title_block_idxes", [])
        paragraph_title_idxes = layout_order_config.get(
            "paragraph_title_block_idxes", []
        )
        # check if there is only one paragraph title and without doc_title
        only_one_paragraph_title = (
            len(paragraph_title_idxes) == 1 and len(doc_title_idxes) == 0
        )
        if only_one_paragraph_title:
            paragraph_title_block_area = caculate_bbox_area(
                layout_det_res["boxes"][paragraph_title_idxes[0]]["coordinate"]
            )
            title_area_max_block_threshold = layout_order_config.get(
                "title_area_max_block_threshold", 0.3
            )
            if (
                paragraph_title_block_area
                > max_block_area * title_area_max_block_threshold
            ):
                layout_det_res["boxes"][paragraph_title_idxes[0]]["label"] = "doc_title"
                layout_order_config["doc_title_block_idxes"].append(
                    paragraph_title_idxes[0]
                )
                layout_order_config["paragraph_title_block_idxes"].remove(
                    paragraph_title_idxes[0]
                )

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
                    x1, y1, x2, y2 = [int(i) for i in crop_box]
                    crop_img = np.array(image)[y1:y2, x1:x2]
                    crop_img_rec_res = next(text_rec_model([crop_img]))
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
                            overall_ocr_res["rec_boxes"] = np.vstack(
                                (overall_ocr_res["rec_boxes"], crop_box)
                            )
                            overall_ocr_res["rec_polys"].append(crop_img_dt_poly)
                            overall_ocr_res["rec_scores"].append(crop_img_rec_score)
                            overall_ocr_res["rec_texts"].append(crop_img_rec_text)
                            overall_ocr_res["rec_labels"].append("text")
                            layout_to_ocr_mapping[box_idx].remove(overall_ocr_idx)
                            layout_to_ocr_mapping[box_idx].append(
                                len(overall_ocr_res["rec_texts"]) - 1
                            )

        layout_order_config["all_layout_region_box"] = region_box
        layout_order_config["layout_to_ocr_mapping"] = layout_to_ocr_mapping
        layout_order_config["matched_ocr_dict"] = matched_ocr_dict

        return layout_order_config, layout_det_res

    def sort_line_by_x_projection(
        self,
        line: List[List[Union[List[int], str]]],
        input_img: np.ndarray,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> None:
        """
        Sort a line of text spans based on their vertical position within the layout bounding box.

        Args:
            line (list): A list of spans, where each span is a list containing a bounding box and text.
            input_img (ndarray): The input image used for OCR.
            general_ocr_pipeline (Any): The general OCR pipeline used for text recognition.

        Returns:
            list: The sorted line of text spans.
        """
        splited_boxes = split_boxes_if_x_contained(line)
        splited_lines = []
        if len(line) != len(splited_boxes):
            splited_boxes.sort(key=lambda span: span[0][0])
            for span in splited_boxes:
                if span[2] == "text":
                    crop_img = input_img[
                        int(span[0][1]) : int(span[0][3]),
                        int(span[0][0]) : int(span[0][2]),
                    ]
                    crop_img_rec_res = next(text_rec_model([crop_img]))
                    crop_img_rec_score = crop_img_rec_res["rec_score"]
                    crop_img_rec_text = crop_img_rec_res["rec_text"]
                    span[1] = (
                        crop_img_rec_text
                        if crop_img_rec_score >= text_rec_score_thresh
                        else ""
                    )

                splited_lines.append(span)
        else:
            splited_lines = line

        return splited_lines

    def get_block_rec_content(
        self,
        image: list,
        layout_order_config: dict,
        ocr_rec_res: dict,
        block: LayoutParsingBlock,
        text_rec_model: Any,
        text_rec_score_thresh: Union[float, None] = None,
    ) -> str:

        text_delimiter_map = {
            "content": "\n",
        }
        line_delimiter_map = {
            "doc_title": " ",
            "content": "\n",
        }
        if len(ocr_rec_res["rec_texts"]) == 0:
            block.content = ""
            return block

        label = block.label
        if label == "reference":
            rec_boxes = ocr_rec_res["boxes"]
            block_left_coordinate = min([box[0] for box in rec_boxes])
            block_right_coordinate = max([box[2] for box in rec_boxes])
            first_line_span_limit = (5,)
            last_line_span_limit = (20,)
        else:
            block_left_coordinate, _, block_right_coordinate, _ = block.bbox
            first_line_span_limit = (10,)
            last_line_span_limit = (10,)

        if label == "formula":
            ocr_rec_res["rec_texts"] = [
                rec_res_text.replace("$", "")
                for rec_res_text in ocr_rec_res["rec_texts"]
            ]
        lines = group_boxes_into_lines(
            ocr_rec_res,
            block,
            layout_order_config.get("line_height_iou_threshold", 0.4),
        )

        block.num_of_lines = len(lines)

        # format line
        new_lines = []
        horizontal_text_line_num = 0
        for line in lines:
            line.sort(key=lambda span: span[0][0])

            # merge formula and text
            ocr_labels = [span[2] for span in line]
            if "formula" in ocr_labels:
                line = self.sort_line_by_x_projection(
                    line, image, text_rec_model, text_rec_score_thresh
                )

            text_orientation = calculate_text_orientation([span[0] for span in line])
            horizontal_text_line_num += 1 if text_orientation == "horizontal" else 0

            line_text = format_line(
                line,
                block_left_coordinate,
                block_right_coordinate,
                first_line_span_limit=first_line_span_limit,
                last_line_span_limit=last_line_span_limit,
                block_label=block.label,
                delimiter_map=text_delimiter_map,
            )
            new_lines.append(line_text)

        delim = line_delimiter_map.get(label, "")
        content = delim.join(new_lines)
        block.content = content
        block.direction = (
            "horizontal"
            if horizontal_text_line_num > len(new_lines) * 0.5
            else "vertical"
        )

        return block

    def get_layout_parsing_blocks(
        self,
        image: list,
        layout_order_config: dict,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
        table_res_list: list,
        seal_res_list: list,
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
        layout_parsing_blocks: List[LayoutParsingBlock] = []

        for box_idx, box_info in enumerate(layout_det_res["boxes"]):

            label = box_info["label"]
            block_bbox = box_info["coordinate"]
            rec_res = {"boxes": [], "rec_texts": [], "rec_labels": []}

            block = LayoutParsingBlock(label=label, bbox=block_bbox)

            if label == "table" and len(table_res_list) > 0:
                block.content = table_res_list[table_index]["pred_html"]
                table_index += 1
            elif label == "seal" and len(seal_res_list) > 0:
                block.content = seal_res_list[seal_index]["rec_texts"]
                seal_index += 1
            else:
                if label == "formula":
                    _, ocr_idx_list = get_sub_regions_ocr_res(
                        overall_ocr_res, [block_bbox], return_match_idx=True
                    )
                    layout_order_config["layout_to_ocr_mapping"][box_idx] = ocr_idx_list
                else:
                    ocr_idx_list = layout_order_config["layout_to_ocr_mapping"].get(
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
                block = self.get_block_rec_content(
                    image=image,
                    block=block,
                    layout_order_config=layout_order_config,
                    ocr_rec_res=rec_res,
                    text_rec_model=text_rec_model,
                    text_rec_score_thresh=text_rec_score_thresh,
                )

            if label in ["chart", "image"]:
                x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                img_path = f"imgs/img_in_table_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                img = Image.fromarray(image[y_min:y_max, x_min:x_max, ::-1])
                block.image = {img_path: img}

            layout_parsing_blocks.append(block)

        # when there is no layout detection result but there is ocr result, use ocr result
        if len(layout_det_res["boxes"]) == 0:
            region_box = [65535, 65535, 0, 0]
            for ocr_idx, (ocr_rec_box, ocr_rec_text) in enumerate(
                zip(overall_ocr_res["rec_boxes"], overall_ocr_res["rec_texts"])
            ):
                update_layout_order_config_block_index(
                    layout_order_config, "text", ocr_idx
                )
                region_box = update_region_box(ocr_rec_box, region_box)
                layout_parsing_blocks.append(
                    LayoutParsingBlock(
                        label="text", bbox=ocr_rec_box, content=ocr_rec_text
                    )
                )
            layout_order_config["all_layout_region_box"] = region_box

        return layout_parsing_blocks, layout_order_config

    def get_layout_parsing_res(
        self,
        image: list,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        table_res_list: list,
        seal_res_list: list,
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
        from .setting import layout_order_config

        # Standardize data
        layout_order_config, layout_det_res = self.standardized_data(
            image=image,
            layout_order_config=copy.deepcopy(layout_order_config),
            layout_det_res=layout_det_res,
            overall_ocr_res=overall_ocr_res,
            formula_res_list=formula_res_list,
            text_rec_model=self.general_ocr_pipeline.text_rec_model,
            text_rec_score_thresh=text_rec_score_thresh,
        )

        # Format layout parsing block
        parsing_res_list, layout_order_config = self.get_layout_parsing_blocks(
            image=image,
            layout_order_config=layout_order_config,
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            table_res_list=table_res_list,
            seal_res_list=seal_res_list,
            text_rec_model=self.general_ocr_pipeline.text_rec_model,
            text_rec_score_thresh=self.general_ocr_pipeline.text_rec_score_thresh,
        )

        parsing_res_list = xycut_enhanced(
            parsing_res_list,
            layout_order_config,
        )

        return parsing_res_list

    def get_model_settings(
        self,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
        use_general_ocr: Union[bool, None],
        use_seal_recognition: Union[bool, None],
        use_table_recognition: Union[bool, None],
        use_formula_recognition: Union[bool, None],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Union[bool, None]): Enables document orientation classification if True. Defaults to system setting if None.
            use_doc_unwarping (Union[bool, None]): Enables document unwarping if True. Defaults to system setting if None.
            use_general_ocr (Union[bool, None]): Enables general OCR if True. Defaults to system setting if None.
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

        if use_general_ocr is None:
            use_general_ocr = self.use_general_ocr

        if use_seal_recognition is None:
            use_seal_recognition = self.use_seal_recognition

        if use_table_recognition is None:
            use_table_recognition = self.use_table_recognition

        if use_formula_recognition is None:
            use_formula_recognition = self.use_formula_recognition

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_general_ocr=use_general_ocr,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
        )

    def predict(
        self,
        input: Union[str, list[str], np.ndarray, list[np.ndarray]],
        use_doc_orientation_classify: Union[bool, None] = None,
        use_doc_unwarping: Union[bool, None] = None,
        use_textline_orientation: Optional[bool] = None,
        use_general_ocr: Union[bool, None] = None,
        use_seal_recognition: Union[bool, None] = None,
        use_table_recognition: Union[bool, None] = None,
        use_formula_recognition: Union[bool, None] = None,
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
        use_table_cells_ocr_results: bool = False,
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
            use_general_ocr (Optional[bool]): Whether to use general OCR.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.
            use_formula_recognition (Optional[bool]): Whether to use formula recognition.
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
            use_table_cells_ocr_results (bool): whether to use OCR results with cells.
            use_e2e_wired_table_rec_model (bool): Whether to use end-to-end wired table recognition model.
            use_e2e_wireless_table_rec_model (bool): Whether to use end-to-end wireless table recognition model.
            **kwargs (Any): Additional settings to extend functionality.

        Returns:
            LayoutParsingResultV2: The predicted layout parsing result.
        """

        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_general_ocr,
            use_seal_recognition,
            use_table_recognition,
            use_formula_recognition,
        )

        if not self.check_model_settings_valid(model_settings):
            yield {"error": "the input params for model settings are invalid!"}

        for batch_data in self.batch_sampler(input):
            image_array = self.img_reader(batch_data.instances)[0]

            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_res = next(
                    self.doc_preprocessor_pipeline(
                        image_array,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    ),
                )
            else:
                doc_preprocessor_res = {"output_img": image_array}

            doc_preprocessor_image = doc_preprocessor_res["output_img"]

            layout_det_res = next(
                self.layout_det_model(
                    doc_preprocessor_image,
                    threshold=layout_threshold,
                    layout_nms=layout_nms,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                )
            )
            imgs_in_doc = gather_imgs(doc_preprocessor_image, layout_det_res["boxes"])

            if model_settings["use_formula_recognition"]:
                formula_res_all = next(
                    self.formula_recognition_pipeline(
                        doc_preprocessor_image,
                        use_layout_detection=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        layout_det_res=layout_det_res,
                    ),
                )
                formula_res_list = formula_res_all["formula_res_list"]
            else:
                formula_res_list = []

            for formula_res in formula_res_list:
                x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = 255.0

            if (
                model_settings["use_general_ocr"]
                or model_settings["use_table_recognition"]
            ):
                overall_ocr_res = next(
                    self.general_ocr_pipeline(
                        doc_preprocessor_image,
                        use_textline_orientation=use_textline_orientation,
                        text_det_limit_side_len=text_det_limit_side_len,
                        text_det_limit_type=text_det_limit_type,
                        text_det_thresh=text_det_thresh,
                        text_det_box_thresh=text_det_box_thresh,
                        text_det_unclip_ratio=text_det_unclip_ratio,
                        text_rec_score_thresh=text_rec_score_thresh,
                    ),
                )
            else:
                overall_ocr_res = {}

            overall_ocr_res["rec_labels"] = ["text"] * len(overall_ocr_res["rec_texts"])

            if model_settings["use_table_recognition"]:
                table_contents = copy.deepcopy(overall_ocr_res)
                for formula_res in formula_res_list:
                    x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                    poly_points = [
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ]
                    table_contents["dt_polys"].append(poly_points)
                    table_contents["rec_texts"].append(
                        f"${formula_res['rec_formula']}$"
                    )
                    table_contents["rec_boxes"] = np.vstack(
                        (table_contents["rec_boxes"], [formula_res["dt_polys"]])
                    )
                    table_contents["rec_polys"].append(poly_points)
                    table_contents["rec_scores"].append(1)

                for img in imgs_in_doc:
                    img_path = img["path"]
                    x_min, y_min, x_max, y_max = img["coordinate"]
                    poly_points = [
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ]
                    table_contents["dt_polys"].append(poly_points)
                    table_contents["rec_texts"].append(
                        f'<div style="text-align: center;"><img src="{img_path}" alt="Image" /></div>'
                    )
                    if table_contents["rec_boxes"].size == 0:
                        table_contents["rec_boxes"] = np.array([img["coordinate"]])
                    else:
                        table_contents["rec_boxes"] = np.vstack(
                            (table_contents["rec_boxes"], img["coordinate"])
                        )
                    table_contents["rec_polys"].append(poly_points)
                    table_contents["rec_scores"].append(img["score"])

                table_res_all = next(
                    self.table_recognition_pipeline(
                        doc_preprocessor_image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        use_ocr_model=False,
                        overall_ocr_res=table_contents,
                        layout_det_res=layout_det_res,
                        cell_sort_by_y_projection=True,
                        use_table_cells_ocr_results=use_table_cells_ocr_results,
                        use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
                        use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
                    ),
                )
                table_res_list = table_res_all["table_res_list"]
            else:
                table_res_list = []

            if model_settings["use_seal_recognition"]:
                seal_res_all = next(
                    self.seal_recognition_pipeline(
                        doc_preprocessor_image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        layout_det_res=layout_det_res,
                        seal_det_limit_side_len=seal_det_limit_side_len,
                        seal_det_limit_type=seal_det_limit_type,
                        seal_det_thresh=seal_det_thresh,
                        seal_det_box_thresh=seal_det_box_thresh,
                        seal_det_unclip_ratio=seal_det_unclip_ratio,
                        seal_rec_score_thresh=seal_rec_score_thresh,
                    ),
                )
                seal_res_list = seal_res_all["seal_res_list"]
            else:
                seal_res_list = []

            parsing_res_list = self.get_layout_parsing_res(
                doc_preprocessor_image,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                table_res_list=table_res_list,
                seal_res_list=seal_res_list,
                formula_res_list=formula_res_list,
                text_rec_score_thresh=text_rec_score_thresh,
            )

            for formula_res in formula_res_list:
                x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
                doc_preprocessor_image[y_min:y_max, x_min:x_max, :] = formula_res[
                    "input_img"
                ]

            single_img_res = {
                "input_path": batch_data.input_paths[0],
                "page_index": batch_data.page_indexes[0],
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "overall_ocr_res": overall_ocr_res,
                "table_res_list": table_res_list,
                "seal_res_list": seal_res_list,
                "formula_res_list": formula_res_list,
                "parsing_res_list": parsing_res_list,
                "imgs_in_doc": imgs_in_doc,
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
