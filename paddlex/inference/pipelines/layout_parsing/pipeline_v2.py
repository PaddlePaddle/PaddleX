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
from __future__ import annotations

from typing import Optional, Union, Tuple
import numpy as np

from ....utils import logging
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...models.object_detection.result import DetResult
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from ..ocr.result import OCRResult
from .result_v2 import LayoutParsingResultV2
from .utils import get_single_block_parsing_res
from .utils import get_sub_regions_ocr_res


class LayoutParsingPipelineV2(BasePipeline):
    """Layout Parsing Pipeline V2"""

    entities = ["layout_parsing_v2"]

    def __init__(
        self,
        config: dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the layout parsing pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """

        super().__init__(
            device=device,
            pp_option=pp_option,
            use_hpip=use_hpip,
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

    def get_layout_parsing_res(
        self,
        image: list,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        table_res_list: list,
        seal_res_list: list,
        formula_res_list: list,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
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
            text_det_limit_side_len (Optional[int], optional): The maximum side length of the text detection region. Defaults to None.
            text_det_limit_type (Optional[str], optional): The type of limit for the text detection region. Defaults to None.
            text_det_thresh (Optional[float], optional): The confidence threshold for text detection. Defaults to None.
            text_det_box_thresh (Optional[float], optional): The confidence threshold for text detection bounding boxes. Defaults to None
            text_det_unclip_ratio (Optional[float], optional): The unclip ratio for text detection. Defaults to None.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.
        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """
        matched_ocr_dict = {}
        image = np.array(image)
        object_boxes = []

        for object_box_idx, box_info in enumerate(layout_det_res["boxes"]):
            box = box_info["coordinate"]
            label = box_info["label"].lower()
            object_boxes.append(box)

            if label not in ["formula", "table", "seal"]:
                _, matched_idxs = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                for matched_idx in matched_idxs:
                    if matched_ocr_dict.get(matched_idx, None) is None:
                        matched_ocr_dict[matched_idx] = [object_box_idx]
                    else:
                        matched_ocr_dict[matched_idx].append(object_box_idx)

        already_processed = set()
        for matched_idx, layout_box_ids in matched_ocr_dict.items():
            if len(layout_box_ids) <= 1:
                continue

            # one ocr is matched to multiple layout boxes, split the text into multiple lines
            for idx in layout_box_ids:
                if idx in already_processed:
                    continue

                already_processed.add(idx)
                wht_im = np.ones(image.shape, dtype=image.dtype) * 255
                box = object_boxes[idx]
                x1, y1, x2, y2 = [int(i) for i in box]
                wht_im[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]
                sub_ocr_res = next(
                    self.general_ocr_pipeline(
                        wht_im,
                        text_det_limit_side_len=text_det_limit_side_len,
                        text_det_limit_type=text_det_limit_type,
                        text_det_thresh=text_det_thresh,
                        text_det_box_thresh=text_det_box_thresh,
                        text_det_unclip_ratio=text_det_unclip_ratio,
                        text_rec_score_thresh=text_rec_score_thresh,
                    )
                )
                _, matched_idxs = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                for matched_idx in sorted(matched_idxs, reverse=True):
                    del overall_ocr_res["dt_polys"][matched_idx]
                    del overall_ocr_res["rec_texts"][matched_idx]
                    overall_ocr_res["rec_boxes"] = np.delete(
                        overall_ocr_res["rec_boxes"], matched_idx, axis=0
                    )
                    del overall_ocr_res["rec_polys"][matched_idx]
                    del overall_ocr_res["rec_scores"][matched_idx]

                if sub_ocr_res["rec_boxes"] != []:
                    overall_ocr_res["dt_polys"].extend(sub_ocr_res["dt_polys"])
                    overall_ocr_res["rec_texts"].extend(sub_ocr_res["rec_texts"])
                    overall_ocr_res["rec_boxes"] = np.concatenate(
                        [overall_ocr_res["rec_boxes"], sub_ocr_res["rec_boxes"]], axis=0
                    )
                    overall_ocr_res["rec_polys"].extend(sub_ocr_res["rec_polys"])
                    overall_ocr_res["rec_scores"].extend(sub_ocr_res["rec_scores"])

        for formula_res in formula_res_list:
            x_min, y_min, x_max, y_max = list(map(int, formula_res["dt_polys"]))
            poly_points = [
                (x_min, y_min),
                (x_max, y_min),
                (x_max, y_max),
                (x_min, y_max),
            ]
            overall_ocr_res["dt_polys"].append(poly_points)
            overall_ocr_res["rec_texts"].append(f"${formula_res['rec_formula']}$")
            overall_ocr_res["rec_boxes"] = np.vstack(
                (overall_ocr_res["rec_boxes"], [formula_res["dt_polys"]])
            )
            overall_ocr_res["rec_polys"].append(poly_points)
            overall_ocr_res["rec_scores"].append(1)

        parsing_res_list = get_single_block_parsing_res(
            overall_ocr_res=overall_ocr_res,
            layout_det_res=layout_det_res,
            table_res_list=table_res_list,
            seal_res_list=seal_res_list,
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
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
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

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
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

            if model_settings["use_table_recognition"]:
                table_res_all = next(
                    self.table_recognition_pipeline(
                        doc_preprocessor_image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        use_ocr_model=False,
                        overall_ocr_res=overall_ocr_res,
                        layout_det_res=layout_det_res,
                        cell_sort_by_y_projection=True,
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
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
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
                "model_settings": model_settings,
            }
            yield LayoutParsingResultV2(single_img_res)
