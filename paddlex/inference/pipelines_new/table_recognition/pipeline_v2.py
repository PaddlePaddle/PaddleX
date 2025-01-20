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

import os, sys
from typing import Any, Dict, Optional
import numpy as np
import cv2
from ..base import BasePipeline
from ..components import CropByBoxes
from .utils import get_neighbor_boxes_idx
from .table_recognition_post_processing_v2 import get_table_recognition_res
from .result import SingleTableRecognitionResult, TableRecognitionResult
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..ocr.result import OCRResult
from ..doc_preprocessor.result import DocPreprocessorResult

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult


class TableRecognitionPipelineV2(BasePipeline):
    """Table Recognition Pipeline"""

    entities = ["table_recognition_v2"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initializes the layout parsing pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            hpi_params (Optional[Dict[str, Any]], optional): HPIP parameters. Defaults to None.
        """

        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_params=hpi_params
        )

        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
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

        self.use_layout_detection = config.get("use_layout_detection", True)
        if self.use_layout_detection:
            layout_det_config = config.get("SubModules", {}).get(
                "LayoutDetection",
                {"model_config_error": "config error for layout_det_model!"},
            )
            self.layout_det_model = self.create_model(layout_det_config)

        table_cls_config = config.get("SubModules", {}).get(
            "TableClassification",
            {"model_config_error": "config error for table_classification_model!"},
        )
        self.table_cls_model = self.create_model(table_cls_config)

        wired_table_rec_config = config.get("SubModules", {}).get(
            "WiredTableStructureRecognition",
            {"model_config_error": "config error for wired_table_structure_model!"},
        )
        self.wired_table_rec_model = self.create_model(wired_table_rec_config)

        wireless_table_rec_config = config.get("SubModules", {}).get(
            "WirelessTableStructureRecognition",
            {"model_config_error": "config error for wireless_table_structure_model!"},
        )
        self.wireless_table_rec_model = self.create_model(wireless_table_rec_config)
        
        wired_table_cells_det_config = config.get("SubModules", {}).get(
            "WiredTableCellsDetection",
            {"model_config_error": "config error for wired_table_cells_detection_model!"},
        )
        self.wired_table_cells_detection_model = self.create_model(wired_table_cells_det_config)

        wireless_table_cells_det_config = config.get("SubModules", {}).get(
            "WirelessTableCellsDetection",
            {"model_config_error": "config error for wireless_table_cells_detection_model!"},
        )
        self.wireless_table_cells_detection_model = self.create_model(wireless_table_cells_det_config)
    
        self.use_ocr_model = config.get("use_ocr_model", True)
        if self.use_ocr_model:
            general_ocr_config = config.get("SubPipelines", {}).get(
                "GeneralOCR",
                {"pipeline_config_error": "config error for general_ocr_pipeline!"},
            )
            self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)

        self._crop_by_boxes = CropByBoxes()

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_layout_detection: Optional[bool],
        use_ocr_model: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_layout_detection (Optional[bool]): Whether to use layout detection.
            use_ocr_model (Optional[bool]): Whether to use OCR model.

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

        if use_layout_detection is None:
            use_layout_detection = self.use_layout_detection

        if use_ocr_model is None:
            use_ocr_model = self.use_ocr_model

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_layout_detection=use_layout_detection,
            use_ocr_model=use_ocr_model,
        )

    def check_model_settings_valid(
        self,
        model_settings: Dict,
        overall_ocr_res: OCRResult,
        layout_det_res: DetResult,
    ) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_settings (Dict): A dictionary containing input parameters.
            overall_ocr_res (OCRResult): Overall OCR result obtained after running the OCR pipeline.
                The overall OCR result with convert_points_to_boxes information.
            layout_det_res (DetResult): The layout detection result.
        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if model_settings["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if model_settings["use_layout_detection"]:
            if layout_det_res is not None:
                logging.error(
                    "The layout detection model has already been initialized, please set use_layout_detection=False"
                )
                return False

            if not self.use_layout_detection:
                logging.error(
                    "Set use_layout_detection, but the models for layout detection are not initialized."
                )
                return False

        if model_settings["use_ocr_model"]:
            if overall_ocr_res is not None:
                logging.error(
                    "The OCR models have already been initialized, please set use_ocr_model=False"
                )
                return False

            if not self.use_ocr_model:
                logging.error(
                    "Set use_ocr_model, but the models for OCR are not initialized."
                )
                return False
        else:
            if overall_ocr_res is None:
                logging.error("Set use_ocr_model=False, but no OCR results were found.")
                return False
        return True

    def predict_doc_preprocessor_res(
        self, image_array: np.ndarray, input_params: dict
    ) -> tuple[DocPreprocessorResult, np.ndarray]:
        """
        Preprocess the document image based on input parameters.

        Args:
            image_array (np.ndarray): The input image array.
            input_params (dict): Dictionary containing preprocessing parameters.

        Returns:
            tuple[DocPreprocessorResult, np.ndarray]: A tuple containing the preprocessing
                                              result dictionary and the processed image array.
        """
        if input_params["use_doc_preprocessor"]:
            use_doc_orientation_classify = input_params["use_doc_orientation_classify"]
            use_doc_unwarping = input_params["use_doc_unwarping"]
            doc_preprocessor_res = next(
                self.doc_preprocessor_pipeline(
                    image_array,
                    use_doc_orientation_classify=use_doc_orientation_classify,
                    use_doc_unwarping=use_doc_unwarping,
                )
            )
            doc_preprocessor_image = doc_preprocessor_res["output_img"]
        else:
            doc_preprocessor_res = {}
            doc_preprocessor_image = image_array
        return doc_preprocessor_res, doc_preprocessor_image

    def extract_results(self, pred, task):
        if task == "cls":
            return pred['label_names'][np.argmax(pred['scores'])]
        elif task == "det":
            threshold = 0.0
            result = []
            if 'boxes' in pred and isinstance(pred['boxes'], list):
                for box in pred['boxes']:
                    if isinstance(box, dict) and 'score' in box and 'coordinate' in box:
                        score = box['score']
                        coordinate = box['coordinate']
                        if isinstance(score, float) and score > threshold:
                            result.append(coordinate)
            return result
        elif task == "table_stru":
            return pred["structure"]
        else:
            return None

    def predict_single_table_recognition_res(
        self,
        image_array: np.ndarray,
        overall_ocr_res: OCRResult,
        table_box: list,
        flag_find_nei_text: bool = True,
    ) -> SingleTableRecognitionResult:
        """
        Predict table recognition results from an image array, layout detection results, and OCR results.

        Args:
            image_array (np.ndarray): The input image represented as a numpy array.
            overall_ocr_res (OCRResult): Overall OCR result obtained after running the OCR pipeline.
                The overall OCR results containing text recognition information.
            table_box (list): The table box coordinates.
            flag_find_nei_text (bool): Whether to find neighboring text.
        Returns:
            SingleTableRecognitionResult: single table recognition result.
        """
        table_cls_pred = next(self.table_cls_model(image_array))
        table_cls_result = self.extract_results(table_cls_pred, "cls")
        if table_cls_result == "wired_table":
            table_structure_pred = next(self.wired_table_rec_model(image_array))
            table_cells_pred = next(self.wired_table_cells_detection_model(image_array))
        elif table_cls_result == "wireless_table":
            table_structure_pred = next(self.wireless_table_rec_model(image_array))
            table_cells_pred = next(self.wireless_table_cells_detection_model(image_array))
        table_structure_result = self.extract_results(table_structure_pred, "table_stru")
        table_cells_result = self.extract_results(table_cells_pred, "det")
        single_table_recognition_res = get_table_recognition_res(
            table_box, table_structure_result, table_cells_result, overall_ocr_res
        )
        neighbor_text = ""
        if flag_find_nei_text:
            match_idx_list = get_neighbor_boxes_idx(
                overall_ocr_res["rec_boxes"], table_box
            )
            if len(match_idx_list) > 0:
                for idx in match_idx_list:
                    neighbor_text += overall_ocr_res["rec_texts"][idx] + "; "
        single_table_recognition_res["neighbor_text"] = neighbor_text
        return single_table_recognition_res

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_layout_detection: Optional[bool] = None,
        use_ocr_model: Optional[bool] = None,
        overall_ocr_res: Optional[OCRResult] = None,
        layout_det_res: Optional[DetResult] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        **kwargs,
    ) -> TableRecognitionResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) of pdf(s) to be processed.
            use_layout_detection (bool): Whether to use layout detection.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            overall_ocr_res (OCRResult): The overall OCR result with convert_points_to_boxes information.
                It will be used if it is not None and use_ocr_model is False.
            layout_det_res (DetResult): The layout detection result.
                It will be used if it is not None and use_layout_detection is False.
            **kwargs: Additional keyword arguments.

        Returns:
            TableRecognitionResult: The predicted table recognition result.
        """

        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_layout_detection,
            use_ocr_model,
        )

        if not self.check_model_settings_valid(
            model_settings, overall_ocr_res, layout_det_res
        ):
            yield {"error": "the input params for model settings are invalid!"}

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            if not isinstance(batch_data[0], str):
                # TODO: add support input_pth for ndarray and pdf
                input_path = f"{img_id}"
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

            if model_settings["use_ocr_model"]:
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

            table_res_list = []
            table_region_id = 1
            if not model_settings["use_layout_detection"] and layout_det_res is None:
                layout_det_res = {}
                img_height, img_width = doc_preprocessor_image.shape[:2]
                table_box = [0, 0, img_width - 1, img_height - 1]
                single_table_rec_res = self.predict_single_table_recognition_res(
                    doc_preprocessor_image,
                    overall_ocr_res,
                    table_box,
                    flag_find_nei_text=False,
                )
                single_table_rec_res["table_region_id"] = table_region_id
                table_res_list.append(single_table_rec_res)
                table_region_id += 1
            else:
                if model_settings["use_layout_detection"]:
                    layout_det_res = next(self.layout_det_model(doc_preprocessor_image))

                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["table"]:
                        crop_img_info = self._crop_by_boxes(image_array, [box_info])
                        crop_img_info = crop_img_info[0]
                        table_box = crop_img_info["box"]
                        single_table_rec_res = (
                            self.predict_single_table_recognition_res(
                                crop_img_info["img"], overall_ocr_res, table_box
                            )
                        )
                        single_table_rec_res["table_region_id"] = table_region_id
                        table_res_list.append(single_table_rec_res)
                        table_region_id += 1

            single_img_res = {
                "input_path": input_path,
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "overall_ocr_res": overall_ocr_res,
                "table_res_list": table_res_list,
                "model_settings": model_settings,
            }
            yield TableRecognitionResult(single_img_res)
