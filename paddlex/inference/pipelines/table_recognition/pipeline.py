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

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...models.object_detection.result import DetResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import CropByBoxes
from ..doc_preprocessor.result import DocPreprocessorResult
from ..ocr.result import OCRResult
from .result import SingleTableRecognitionResult, TableRecognitionResult
from .table_recognition_post_processing import get_table_recognition_res
from .utils import get_neighbor_boxes_idx


class _TableRecognitionPipeline(BasePipeline):
    """Table Recognition Pipeline"""

    def __init__(
        self,
        config: Dict,
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
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
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

        table_structure_config = config.get("SubModules", {}).get(
            "TableStructureRecognition",
            {"model_config_error": "config error for table_structure_model!"},
        )
        self.table_structure_model = self.create_model(table_structure_config)

        self.use_ocr_model = config.get("use_ocr_model", True)
        if self.use_ocr_model:
            general_ocr_config = config.get("SubPipelines", {}).get(
                "GeneralOCR",
                {"pipeline_config_error": "config error for general_ocr_pipeline!"},
            )
            self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)
        else:
            self.general_ocr_config_bak = config.get("SubPipelines", {}).get(
                "GeneralOCR", None
            )

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
    ) -> Tuple[DocPreprocessorResult, np.ndarray]:
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

    def split_ocr_bboxes_by_table_cells(self, ori_img, cells_bboxes):
        """
        Splits OCR bounding boxes by table cells and retrieves text.

        Args:
            ori_img (ndarray): The original image from which text regions will be extracted.
            cells_bboxes (list or ndarray): Detected cell bounding boxes to extract text from.

        Returns:
            list: A list containing the recognized texts from each cell.
        """

        # Check if cells_bboxes is a list and convert it if not.
        if not isinstance(cells_bboxes, list):
            cells_bboxes = cells_bboxes.tolist()
        texts_list = []  # Initialize a list to store the recognized texts.
        # Process each bounding box provided in cells_bboxes.
        for i in range(len(cells_bboxes)):
            # Extract and round up the coordinates of the bounding box.
            x1, y1, x2, y2 = [math.ceil(k) for k in cells_bboxes[i]]
            # Perform OCR on the defined region of the image and get the recognized text.
            rec_te = next(self.general_ocr_pipeline(ori_img[y1:y2, x1:x2, :]))
            # Concatenate the texts and append them to the texts_list.
            texts_list.append("".join(rec_te["rec_texts"]))
        # Return the list of recognized texts from each cell.
        return texts_list

    def split_ocr_bboxes_by_table_cells(self, ori_img, cells_bboxes):
        """
        Splits OCR bounding boxes by table cells and retrieves text.

        Args:
            ori_img (ndarray): The original image from which text regions will be extracted.
            cells_bboxes (list or ndarray): Detected cell bounding boxes to extract text from.

        Returns:
            list: A list containing the recognized texts from each cell.
        """

        # Check if cells_bboxes is a list and convert it if not.
        if not isinstance(cells_bboxes, list):
            cells_bboxes = cells_bboxes.tolist()
        texts_list = []  # Initialize a list to store the recognized texts.
        # Process each bounding box provided in cells_bboxes.
        for i in range(len(cells_bboxes)):
            # Extract and round up the coordinates of the bounding box.
            x1, y1, x2, y2 = [math.ceil(k) for k in cells_bboxes[i]]
            # Perform OCR on the defined region of the image and get the recognized text.
            rec_te = next(self.general_ocr_pipeline(ori_img[y1:y2, x1:x2, :]))
            # Concatenate the texts and append them to the texts_list.
            texts_list.append("".join(rec_te["rec_texts"]))
        # Return the list of recognized texts from each cell.
        return texts_list

    def predict_single_table_recognition_res(
        self,
        image_array: np.ndarray,
        overall_ocr_res: OCRResult,
        table_box: list,
        use_ocr_results_with_table_cells: bool = False,
        flag_find_nei_text: bool = True,
        cell_sort_by_y_projection: bool = False,
    ) -> SingleTableRecognitionResult:
        """
        Predict table recognition results from an image array, layout detection results, and OCR results.

        Args:
            image_array (np.ndarray): The input image represented as a numpy array.
            overall_ocr_res (OCRResult): Overall OCR result obtained after running the OCR pipeline.
                The overall OCR results containing text recognition information.
            table_box (list): The table box coordinates.
            use_ocr_results_with_table_cells (bool): whether to use OCR results with cells.
            flag_find_nei_text (bool): Whether to find neighboring text.
            cell_sort_by_y_projection (bool): Whether to sort the matched OCR boxes by y-projection.
        Returns:
            SingleTableRecognitionResult: single table recognition result.
        """
        table_structure_pred = next(self.table_structure_model(image_array))
        if use_ocr_results_with_table_cells == True:
            table_cells_result = table_structure_pred["bbox"]
            table_cells_result = [
                [rect[0], rect[1], rect[4], rect[5]] for rect in table_cells_result
            ]
            cells_texts_list = self.split_ocr_bboxes_by_table_cells(
                image_array, table_cells_result
            )
        else:
            cells_texts_list = []
        single_table_recognition_res = get_table_recognition_res(
            table_box,
            table_structure_pred,
            overall_ocr_res,
            cells_texts_list,
            use_ocr_results_with_table_cells,
            cell_sort_by_y_projection=cell_sort_by_y_projection,
        )
        neighbor_text = ""
        if flag_find_nei_text:
            match_idx_list = get_neighbor_boxes_idx(
                overall_ocr_res["rec_boxes"], table_box
            )
            if len(match_idx_list) > 0:
                for idx in match_idx_list:
                    neighbor_text += overall_ocr_res["rec_texts"][idx] + "; "
        single_table_recognition_res["neighbor_texts"] = neighbor_text
        return single_table_recognition_res

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
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
        use_ocr_results_with_table_cells: bool = False,
        cell_sort_by_y_projection: Optional[bool] = None,
        **kwargs,
    ) -> TableRecognitionResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) of pdf(s) to be processed.
            use_layout_detection (bool): Whether to use layout detection.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            overall_ocr_res (OCRResult): The overall OCR result with convert_points_to_boxes information.
                It will be used if it is not None and use_ocr_model is False.
            layout_det_res (DetResult): The layout detection result.
                It will be used if it is not None and use_layout_detection is False.
            use_ocr_results_with_table_cells (bool): whether to use OCR results with cells.
            cell_sort_by_y_projection (bool): Whether to sort the matched OCR boxes by y-projection.
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
        if cell_sort_by_y_projection is None:
            cell_sort_by_y_projection = False

        if not self.check_model_settings_valid(
            model_settings, overall_ocr_res, layout_det_res
        ):
            yield {"error": "the input params for model settings are invalid!"}

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data.instances)[0]

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
            elif use_ocr_results_with_table_cells == True:
                assert self.general_ocr_config_bak != None
                self.general_ocr_pipeline = self.create_pipeline(
                    self.general_ocr_config_bak
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
                    use_ocr_results_with_table_cells,
                    flag_find_nei_text=False,
                    cell_sort_by_y_projection=cell_sort_by_y_projection,
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
                                crop_img_info["img"],
                                overall_ocr_res,
                                table_box,
                                use_ocr_results_with_table_cells,
                                cell_sort_by_y_projection=cell_sort_by_y_projection,
                            )
                        )
                        single_table_rec_res["table_region_id"] = table_region_id
                        table_res_list.append(single_table_rec_res)
                        table_region_id += 1

            single_img_res = {
                "input_path": batch_data.input_paths[0],
                "page_index": batch_data.page_indexes[0],
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "overall_ocr_res": overall_ocr_res,
                "table_res_list": table_res_list,
                "model_settings": model_settings,
            }
            yield TableRecognitionResult(single_img_res)


@pipeline_requires_extra("ocr")
class TableRecognitionPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = ["table_recognition"]

    @property
    def _pipeline_cls(self):
        return _TableRecognitionPipeline

    def _get_batch_size(self, config):
        return 1
