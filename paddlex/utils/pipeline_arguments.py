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

from ast import literal_eval
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import TypeAdapter, ValidationError


def custom_type(cli_expected_type):
    """Create validator for CLI input conversion and type checking"""

    def validator(cli_input: str) -> cli_expected_type:
        try:
            parsed = literal_eval(cli_input)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
            err = f"""Malformed input:
            - Input: {cli_input!r}
            - Error: {exc}"""
            raise ValueError(err) from exc

        try:
            return TypeAdapter(cli_expected_type).validate_python(parsed)
        except ValidationError as exc:
            err = f"""Invalid input type:
            - Expected: {cli_expected_type}
            - Received: {cli_input!r}
            """
            raise ValueError(err) from exc

    return validator


PIPELINE_ARGUMENTS = {
    "OCR": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_textline_orientation",
            "type": bool,
            "help": "Determines whether to consider text line orientation",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "object_detection": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Sets the threshold for object detection.",
        },
    ],
    "image_classification": [
        {
            "name": "--topk",
            "type": int,
            "help": "Sets the Top-K value for image classification.",
        },
    ],
    "image_multilabel_classification": [
        {
            "name": "--threshold",
            "type": float,
            "help": "Sets the threshold for image multilabel classification.",
        },
    ],
    "pedestrian_attribute_recognition": [
        {
            "name": "--det_threshold",
            "type": float,
            "help": "Sets the threshold for human detection.",
        },
        {
            "name": "--cls_threshold",
            "type": float,
            "help": "Sets the threshold for pedestrian attribute recognition.",
        },
    ],
    "vehicle_attribute_recognition": [
        {
            "name": "--det_threshold",
            "type": float,
            "help": "Sets the threshold for vehicle detection.",
        },
        {
            "name": "--cls_threshold",
            "type": float,
            "help": "Sets the threshold for vehicle attribute recognition.",
        },
    ],
    "human_keypoint_detection": [
        {
            "name": "--det_threshold",
            "type": custom_type(Optional[float]),
            "help": "Sets the threshold for human detection.",
        },
    ],
    "table_recognition": [
        {
            "name": "--use_table_cells_ocr_results",
            "type": bool,
            "help": "Determines whether to use cells OCR results",
        },
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document preprocessing",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_layout_detection",
            "type": bool,
            "help": "Determines whether to use document layout detection",
        },
        {
            "name": "--use_ocr_model",
            "type": bool,
            "help": "Determines whether to use OCR",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "table_recognition_v2": [
        {
            "name": "--use_table_cells_ocr_results",
            "type": bool,
            "help": "Determines whether to use cells OCR results",
        },
        {
            "name": "--use_e2e_wired_table_rec_model",
            "type": bool,
            "help": "Determines whether to use end-to-end wired table recognition model",
        },
        {
            "name": "--use_e2e_wireless_table_rec_model",
            "type": bool,
            "help": "Determines whether to use end-to-end wireless table recognition model",
        },
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document preprocessing",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_layout_detection",
            "type": bool,
            "help": "Determines whether to use document layout detection",
        },
        {
            "name": "--use_ocr_model",
            "type": bool,
            "help": "Determines whether to use OCR",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "seal_recognition": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document preprocessing",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_layout_detection",
            "type": bool,
            "help": "Determines whether to use document layout detection",
        },
        {
            "name": "--layout_threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Determines confidence threshold for layout detection",
        },
        {
            "name": "--layout_nms",
            "type": bool,
            "help": "Determines whether to use non maximum suppression",
        },
        {
            "name": "--layout_unclip_ratio",
            "type": custom_type(
                Optional[Union[float, Tuple[float, float], Dict[int, Tuple]]]
            ),
            "help": "Determines unclip ratio for layout detection boxes",
        },
        {
            "name": "--layout_merge_bboxes_mode",
            "type": custom_type(Optional[Union[str, Dict[int, str]]]),
            "help": "Determines merge mode for layout detection bboxes, 'union', 'large' or 'small'",
        },
        {
            "name": "--seal_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--seal_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection, 'min', 'max'.",
        },
        {
            "name": "--seal_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--seal_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--seal_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--seal_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "layout_parsing": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_general_ocr",
            "type": bool,
            "help": "Determines whether to use general ocr",
        },
        {
            "name": "--use_textline_orientation",
            "type": bool,
            "help": "Determines whether to consider text line orientation",
        },
        {
            "name": "--use_seal_recognition",
            "type": bool,
            "help": "Determines whether to use seal recognition",
        },
        {
            "name": "--use_table_recognition",
            "type": bool,
            "help": "Determines whether to use table recognition",
        },
        {
            "name": "--use_formula_recognition",
            "type": bool,
            "help": "Determines whether to use formula recognition",
        },
        {
            "name": "--layout_threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Determines confidence threshold for layout detection",
        },
        {
            "name": "--layout_nms",
            "type": bool,
            "help": "Determines whether to use non maximum suppression",
        },
        {
            "name": "--layout_unclip_ratio",
            "type": custom_type(
                Optional[Union[float, Tuple[float, float], Dict[int, Tuple]]]
            ),
            "help": "Determines unclip ratio for layout detection boxes",
        },
        {
            "name": "--layout_merge_bboxes_mode",
            "type": custom_type(Optional[Union[str, Dict[int, str]]]),
            "help": "Determines merge mode for layout detection bboxes, 'union', 'large' or 'small'",
        },
        {
            "name": "--seal_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--seal_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection, 'min', 'max'.",
        },
        {
            "name": "--seal_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--seal_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--seal_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--seal_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "PP-StructureV3": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_general_ocr",
            "type": bool,
            "help": "Determines whether to use general ocr",
        },
        {
            "name": "--use_textline_orientation",
            "type": bool,
            "help": "Determines whether to consider text line orientation",
        },
        {
            "name": "--use_seal_recognition",
            "type": bool,
            "help": "Determines whether to use seal recognition",
        },
        {
            "name": "--use_table_recognition",
            "type": bool,
            "help": "Determines whether to use table recognition",
        },
        {
            "name": "--use_formula_recognition",
            "type": bool,
            "help": "Determines whether to use formula recognition",
        },
        {
            "name": "--layout_threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Determines confidence threshold for layout detection",
        },
        {
            "name": "--layout_nms",
            "type": bool,
            "help": "Determines whether to use non maximum suppression",
        },
        {
            "name": "--layout_unclip_ratio",
            "type": custom_type(
                Optional[Union[float, Tuple[float, float], Dict[int, Tuple]]]
            ),
            "help": "Determines unclip ratio for layout detection boxes",
        },
        {
            "name": "--layout_merge_bboxes_mode",
            "type": custom_type(Optional[Union[str, Dict[int, str]]]),
            "help": "Determines merge mode for layout detection bboxes, 'union', 'large' or 'small'",
        },
        {
            "name": "--seal_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--seal_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection, 'min', 'max'.",
        },
        {
            "name": "--seal_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--seal_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--seal_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--seal_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
        {
            "name": "--use_table_cells_ocr_results",
            "type": bool,
            "help": "Determines whether to use cells OCR results",
        },
        {
            "name": "--use_e2e_wired_table_rec_model",
            "type": bool,
            "help": "Determines whether to use end-to-end wired table recognition model",
        },
        {
            "name": "--use_e2e_wireless_table_rec_model",
            "type": bool,
            "help": "Determines whether to use end-to-end wireless table recognition model",
        },
    ],
    "ts_forecast": None,
    "ts_anomaly_detection": None,
    "ts_classification": None,
    "formula_recognition": [
        {
            "name": "--use_layout_detection",
            "type": bool,
            "help": "Determines whether to use layout detection",
        },
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--layout_threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Sets the layout threshold for layout detection.",
        },
        {
            "name": "--layout_nms",
            "type": bool,
            "help": "Determines whether to use layout nms",
        },
        {
            "name": "--layout_unclip_ratio",
            "type": custom_type(
                Optional[Union[float, Tuple[float, float], Dict[int, Tuple]]]
            ),
            "help": "Sets the layout unclip ratio for layout detection.",
        },
        {
            "name": "--layout_merge_bboxes_mode",
            "type": custom_type(Optional[Union[str, Dict[int, str]]]),
            "help": "Sets the layout merge bboxes mode for layout detection.",
        },
    ],
    "instance_segmentation": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[float]),
            "help": "Sets the threshold for instance segmentation.",
        },
    ],
    "semantic_segmentation": [
        {
            "name": "--target_size",
            "type": custom_type(Optional[Union[int, Tuple[int, int], Literal[-1]]]),
            "help": "Sets the inference image resolution for semantic segmentation.",
        },
    ],
    "small_object_detection": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Sets the threshold for small object detection.",
        },
    ],
    "anomaly_detection": None,
    "video_classification": [
        {
            "name": "--topk",
            "type": int,
            "help": "Sets the Top-K value for video classification.",
        },
    ],
    "video_detection": [
        {
            "name": "--nms_thresh",
            "type": float,
            "help": "Sets the NMS threshold for video detection.",
        },
        {
            "name": "--score_thresh",
            "type": float,
            "help": "Sets the confidence threshold for video detection.",
        },
    ],
    "doc_preprocessor": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification.",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping.",
        },
    ],
    "rotated_object_detection": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[Union[float, Dict[int, float]]]),
            "help": "Sets the threshold for rotated object detection.",
        },
    ],
    "open_vocabulary_detection": [
        {
            "name": "--thresholds",
            "type": custom_type(Dict[str, float]),
            "help": "Sets the thresholds for open vocabulary detection.",
        },
        {
            "name": "--prompt",
            "type": str,
            "help": "Sets the prompt for open vocabulary detection.",
        },
    ],
    "open_vocabulary_segmentation": [
        {
            "name": "--prompt_type",
            "type": str,
            "help": "Sets the prompt type for open vocabulary segmentation.",
        },
        {
            "name": "--prompt",
            "type": custom_type(List[List[float]]),
            "help": "Sets the prompt for open vocabulary segmentation.",
        },
    ],
    "3d_bev_detection": None,
    "multilingual_speech_recognition": None,
}
