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

from typing import Any, Dict, Optional
import os, sys
import numpy as np
import cv2
from ..base import BasePipeline
from .utils import get_sub_regions_ocr_res
from ..components import convert_points_to_boxes
from .result import LayoutParsingResult
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..ocr.result import OCRResult

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult


class LayoutParsingPipeline(BasePipeline):
    """Layout Parsing Pipeline"""

    entities = ["layout_parsing"]

    def __init__(
        self,
        config: Dict,
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

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        self.inintial_predictor(config)

        self.batch_sampler = ImageBatchSampler(batch_size=1)

        self.img_reader = ReadImage(format="BGR")

    def inintial_predictor(self, config: Dict) -> None:
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
        self.use_formula_recognition = config.get("use_formula_recognition", True)

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!"
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        layout_det_config = config.get("SubModules", {}).get(
            "LayoutDetection",
            {"model_config_error": "config error for layout_det_model!"},
        )
        self.layout_det_model = self.create_model(layout_det_config)

        if self.use_general_ocr or self.use_table_recognition:
            general_ocr_config = config.get("SubPipelines", {}).get(
                "GeneralOCR",
                {"pipeline_config_error": "config error for general_ocr_pipeline!"},
            )
            self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)

        if self.use_seal_recognition:
            seal_recognition_config = config.get("SubPipelines", {}).get(
                "SealRecognition",
                {
                    "pipeline_config_error": "config error for seal_recognition_pipeline!"
                },
            )
            self.seal_recognition_pipeline = self.create_pipeline(
                seal_recognition_config
            )

        if self.use_table_recognition:
            table_recognition_config = config.get("SubPipelines", {}).get(
                "TableRecognition",
                {
                    "pipeline_config_error": "config error for table_recognition_pipeline!"
                },
            )
            self.table_recognition_pipeline = self.create_pipeline(
                table_recognition_config
            )

        if self.use_formula_recognition:
            formula_recognition_config = config.get("SubPipelines", {}).get(
                "FormulaRecognition",
                {
                    "pipeline_config_error": "config error for formula_recognition_pipeline!"
                },
            )
            self.formula_recognition_pipeline = self.create_pipeline(
                formula_recognition_config
            )

        return

    def get_text_paragraphs_ocr_res(
        self, overall_ocr_res: OCRResult, layout_det_res: DetResult
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
        return get_sub_regions_ocr_res(overall_ocr_res, object_boxes, flag_within=False)

    def check_model_settings_valid(self, input_params: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if input_params["use_general_ocr"] and not self.use_general_ocr:
            logging.error(
                "Set use_general_ocr, but the models for general OCR are not initialized."
            )
            return False

        if input_params["use_seal_recognition"] and not self.use_seal_recognition:
            logging.error(
                "Set use_seal_recognition, but the models for seal recognition are not initialized."
            )
            return False

        if input_params["use_table_recognition"] and not self.use_table_recognition:
            logging.error(
                "Set use_table_recognition, but the models for table recognition are not initialized."
            )
            return False

        return True

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_general_ocr: Optional[bool],
        use_seal_recognition: Optional[bool],
        use_table_recognition: Optional[bool],
        use_formula_recognition: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_general_ocr (Optional[bool]): Whether to use general OCR.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.

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
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_general_ocr: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        **kwargs,
    ) -> LayoutParsingResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or pdf(s) to be processed.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            use_general_ocr (bool): Whether to use general OCR.
            use_seal_recognition (bool): Whether to use seal recognition.
            use_table_recognition (bool): Whether to use table recognition.
            **kwargs: Additional keyword arguments.

        Returns:
            LayoutParsingResult: The predicted layout parsing result.
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
            if not isinstance(batch_data[0], str):
                # TODO: add support input_pth for ndarray and pdf
                input_path = f"{img_id}.jpg"
            else:
                input_path = batch_data[0]

            image_array = self.img_reader(batch_data)[0]

            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_res = next(
                    self.doc_preprocessor_pipeline(
                        image_array,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
            else:
                doc_preprocessor_res = {"output_img": image_array}

            doc_preprocessor_image = doc_preprocessor_res["output_img"]

            layout_det_res = next(self.layout_det_model(doc_preprocessor_image))

            if (
                model_settings["use_general_ocr"]
                or model_settings["use_table_recognition"]
            ):
                overall_ocr_res = next(
                    self.general_ocr_pipeline(
                        doc_preprocessor_image,
                        text_det_limit_side_len=text_det_limit_side_len,
                        text_det_limit_type=text_det_limit_type,
                        text_det_thresh=text_det_thresh,
                        text_det_box_thresh=text_det_box_thresh,
                        text_det_unclip_ratio=text_det_unclip_ratio,
                        text_rec_score_thresh=text_rec_score_thresh,
                    )
                )
            else:
                overall_ocr_res = {}

            if model_settings["use_general_ocr"]:
                text_paragraphs_ocr_res = self.get_text_paragraphs_ocr_res(
                    overall_ocr_res, layout_det_res
                )
            else:
                text_paragraphs_ocr_res = {}

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
                    )
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
                    )
                )
                seal_res_list = seal_res_all["seal_res_list"]
            else:
                seal_res_list = []

            if model_settings["use_formula_recognition"]:
                formula_res_all = next(
                    self.formula_recognition_pipeline(
                        doc_preprocessor_image,
                        use_layout_detection=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        layout_det_res=layout_det_res,
                    )
                )
                formula_res_list = formula_res_all["formula_res_list"]
                print(formula_res_list)
            else:
                formula_res_list = []

            single_img_res = {
                "input_path": input_path,
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "overall_ocr_res": overall_ocr_res,
                "text_paragraphs_ocr_res": text_paragraphs_ocr_res,
                "table_res_list": table_res_list,
                "seal_res_list": seal_res_list,
                "formula_res_list": formula_res_list,
                "model_settings": model_settings,
            }
            yield LayoutParsingResult(single_img_res)
