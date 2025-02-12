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
from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans
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

from ...models.object_detection.result import DetResult


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
            {
                "model_config_error": "config error for wired_table_cells_detection_model!"
            },
        )
        self.wired_table_cells_detection_model = self.create_model(
            wired_table_cells_det_config
        )

        wireless_table_cells_det_config = config.get("SubModules", {}).get(
            "WirelessTableCellsDetection",
            {
                "model_config_error": "config error for wireless_table_cells_detection_model!"
            },
        )
        self.wireless_table_cells_detection_model = self.create_model(
            wireless_table_cells_det_config
        )

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

    def extract_results(self, pred, task):
        if task == "cls":
            return pred["label_names"][np.argmax(pred["scores"])]
        elif task == "det":
            threshold = 0.0
            result = []
            cell_score = []
            if "boxes" in pred and isinstance(pred["boxes"], list):
                for box in pred["boxes"]:
                    if isinstance(box, dict) and "score" in box and "coordinate" in box:
                        score = box["score"]
                        coordinate = box["coordinate"]
                        if isinstance(score, float) and score > threshold:
                            result.append(coordinate)
                            cell_score.append(score)
            return result, cell_score
        elif task == "table_stru":
            return pred["structure"]
        else:
            return None
    
    def cells_det_results_nms(self, cells_det_results, cells_det_scores, cells_det_threshold=0.3):
        """
        Apply Non-Maximum Suppression (NMS) on detection results to remove redundant overlapping bounding boxes.

        Args:
            cells_det_results (list): List of bounding boxes, each box is in format [x1, y1, x2, y2].
            cells_det_scores (list): List of confidence scores corresponding to the bounding boxes.
            cells_det_threshold (float): IoU threshold for suppression. Boxes with IoU greater than this threshold
                                        will be suppressed. Default is 0.5.

        Returns:
        Tuple[list, list]: A tuple containing the list of bounding boxes and confidence scores after NMS,
                            while maintaining one-to-one correspondence.
        """
        # Convert lists to numpy arrays for efficient computation
        boxes = np.array(cells_det_results)
        scores = np.array(cells_det_scores)
        # Initialize list for picked indices
        picked_indices = []
        # Get coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # Compute the area of the bounding boxes
        areas = (x2 - x1) * (y2 - y1)
        # Sort the bounding boxes by the confidence scores in descending order
        order = scores.argsort()[::-1]
        # Process the boxes
        while order.size > 0:
            # Index of the current highest score box
            i = order[0]
            picked_indices.append(i)
            # Compute IoU between the highest score box and the rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # Compute the width and height of the overlapping area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            # Compute the ratio of overlap (IoU)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # Indices of boxes with IoU less than threshold
            inds = np.where(ovr <= cells_det_threshold)[0]
            # Update order, only keep boxes with IoU less than threshold
            order = order[inds + 1]  # inds shifted by 1 because order[0] is the current box
        # Select the boxes and scores based on picked indices
        final_boxes = boxes[picked_indices].tolist()
        final_scores = scores[picked_indices].tolist()
        return final_boxes, final_scores
    
    def get_region_ocr_det_boxes(self, ocr_det_boxes, table_box):
        """Adjust the coordinates of ocr_det_boxes that are fully inside table_box relative to table_box.

        Args:
            ocr_det_boxes (list of list): List of bounding boxes [x1, y1, x2, y2] in the original image.
            table_box (list): Bounding box [x1, y1, x2, y2] of the target region in the original image.

        Returns:
            list of list: List of adjusted bounding boxes relative to table_box, for boxes fully inside table_box.
        """
        tol=0
        # Extract coordinates from table_box
        x_min_t, y_min_t, x_max_t, y_max_t = table_box
        adjusted_boxes = []
        for box in ocr_det_boxes:
            x_min_b, y_min_b, x_max_b, y_max_b = box
            # Check if the box is fully inside table_box
            if (x_min_b+tol >= x_min_t and y_min_b+tol >= y_min_t and
                x_max_b-tol <= x_max_t and y_max_b-tol <= y_max_t):
                # Adjust the coordinates to be relative to table_box
                adjusted_box = [
                    x_min_b - x_min_t,  # Adjust x1
                    y_min_b - y_min_t,  # Adjust y1
                    x_max_b - x_min_t,  # Adjust x2
                    y_max_b - y_min_t   # Adjust y2
                ]
                adjusted_boxes.append(adjusted_box)
            # Discard boxes not fully inside table_box
        return adjusted_boxes

    def cells_det_results_reprocessing(self, cells_det_results, cells_det_scores, ocr_det_results, html_pred_boxes_nums):
        """
        Process and filter cells_det_results based on ocr_det_results and html_pred_boxes_nums.

        Args:
            cells_det_results (List[List[float]]): List of detected cell rectangles [[x1, y1, x2, y2], ...].
            cells_det_scores (List[float]): List of confidence scores for each rectangle in cells_det_results.
            ocr_det_results (List[List[float]]): List of OCR detected rectangles [[x1, y1, x2, y2], ...].
            html_pred_boxes_nums (int): The desired number of rectangles in the final output.

        Returns:
            List[List[float]]: The processed list of rectangles.
        """
        # Function to compute IoU between two rectangles
        def compute_iou(box1, box2):
            """
            Compute the Intersection over Union (IoU) between two rectangles.

            Args:
                box1 (array-like): [x1, y1, x2, y2] of the first rectangle.
                box2 (array-like): [x1, y1, x2, y2] of the second rectangle.

            Returns:
                float: The IoU between the two rectangles.
            """
            # Determine the coordinates of the intersection rectangle
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            # Calculate the area of intersection rectangle
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            # Calculate the area of both rectangles
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            # Calculate the IoU
            iou = intersection_area / float(box1_area)
            return iou

        # Function to combine rectangles into N rectangles
        def combine_rectangles(rectangles, N):
            """
            Combine rectangles into N rectangles based on geometric proximity.

            Args:
                rectangles (list of list of int): A list of rectangles, each represented by [x1, y1, x2, y2].
                N (int): The desired number of combined rectangles.

            Returns:
                list of list of int: A list of N combined rectangles.
            """
            # Number of input rectangles
            num_rects = len(rectangles)
            # If N is greater than or equal to the number of rectangles, return the original rectangles
            if N >= num_rects:
                return rectangles
            # Compute the center points of the rectangles
            centers = np.array([
                [
                    (rect[0] + rect[2]) / 2,  # Center x-coordinate
                    (rect[1] + rect[3]) / 2   # Center y-coordinate
                ]
                for rect in rectangles
            ])
            # Perform KMeans clustering on the center points to group them into N clusters
            kmeans = KMeans(n_clusters=N, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(centers)
            # Initialize a list to store the combined rectangles
            combined_rectangles = []
            # For each cluster, compute the minimal bounding rectangle that covers all rectangles in the cluster
            for i in range(N):
                # Get the indices of rectangles that belong to cluster i
                indices = np.where(labels == i)[0]
                if len(indices) == 0:
                    # If no rectangles in this cluster, skip it
                    continue
                # Extract the rectangles in cluster i
                cluster_rects = np.array([rectangles[idx] for idx in indices])
                # Compute the minimal x1, y1 (top-left corner) and maximal x2, y2 (bottom-right corner)
                x1_min = np.min(cluster_rects[:, 0])
                y1_min = np.min(cluster_rects[:, 1])
                x2_max = np.max(cluster_rects[:, 2])
                y2_max = np.max(cluster_rects[:, 3])
                # Append the combined rectangle to the list
                combined_rectangles.append([x1_min, y1_min, x2_max, y2_max])
            return combined_rectangles

        # Ensure that the inputs are numpy arrays for efficient computation
        cells_det_results = np.array(cells_det_results)
        cells_det_scores = np.array(cells_det_scores)
        ocr_det_results = np.array(ocr_det_results)
        more_cells_flag = False
        if len(cells_det_results) == html_pred_boxes_nums:
            return cells_det_results
        # Step 1: If cells_det_results has more rectangles than html_pred_boxes_nums
        elif len(cells_det_results) > html_pred_boxes_nums:
            more_cells_flag = True
            # Select the indices of the top html_pred_boxes_nums scores
            top_indices = np.argsort(-cells_det_scores)[:html_pred_boxes_nums]
            # Adjust the corresponding rectangles
            cells_det_results = cells_det_results[top_indices].tolist()
        # Threshold for IoU
        iou_threshold = 0.6
        # List to store ocr_miss_boxes
        ocr_miss_boxes = []
        # For each rectangle in ocr_det_results
        for ocr_rect in ocr_det_results:
            merge_ocr_box_iou = []
            # Flag to indicate if ocr_rect has IoU >= threshold with any cell_rect
            has_large_iou = False
            # For each rectangle in cells_det_results
            for cell_rect in cells_det_results:
                # Compute IoU
                iou = compute_iou(ocr_rect, cell_rect)
                if iou > 0:
                    merge_ocr_box_iou.append(iou)
                if (iou>=iou_threshold) or (sum(merge_ocr_box_iou)>=iou_threshold):
                    has_large_iou = True
                    break
            if not has_large_iou:
                ocr_miss_boxes.append(ocr_rect)
        # If no ocr_miss_boxes, return cells_det_results
        if len(ocr_miss_boxes) == 0:
            final_results = cells_det_results if more_cells_flag==True else cells_det_results.tolist()
        else:
            if more_cells_flag == True:
                final_results = combine_rectangles(cells_det_results+ocr_miss_boxes, html_pred_boxes_nums)
            else:
                # Need to combine ocr_miss_boxes into N rectangles
                N = html_pred_boxes_nums - len(cells_det_results)
                # Combine ocr_miss_boxes into N rectangles
                ocr_supp_boxes = combine_rectangles(ocr_miss_boxes, N)
                # Combine cells_det_results and ocr_supp_boxes
                final_results = np.concatenate((cells_det_results, ocr_supp_boxes), axis=0).tolist()
        if len(final_results) <= 0.6*html_pred_boxes_nums:
            final_results = combine_rectangles(ocr_det_results, html_pred_boxes_nums)
        return final_results

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
            table_cells_pred = next(
                self.wired_table_cells_detection_model(image_array, threshold=0.3)
            ) # Setting the threshold to 0.3 can improve the accuracy of table cells detection. 
              # If you really want more or fewer table cells detection boxes, the threshold can be adjusted.
        elif table_cls_result == "wireless_table":
            table_structure_pred = next(self.wireless_table_rec_model(image_array))
            table_cells_pred = next(
                self.wireless_table_cells_detection_model(image_array, threshold=0.3)
            ) # Setting the threshold to 0.3 can improve the accuracy of table cells detection. 
              # If you really want more or fewer table cells detection boxes, the threshold can be adjusted.
        table_structure_result = self.extract_results(table_structure_pred, "table_stru")
        table_cells_result, table_cells_score = self.extract_results(table_cells_pred, "det")
        table_cells_result, table_cells_score = self.cells_det_results_nms(table_cells_result, table_cells_score)
        ocr_det_boxes = self.get_region_ocr_det_boxes(overall_ocr_res["rec_boxes"].tolist(), table_box)
        table_cells_result = self.cells_det_results_reprocessing(
            table_cells_result, table_cells_score, ocr_det_boxes, len(table_structure_pred['bbox'])
        )
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
                "input_path": batch_data.input_paths[0],
                "page_index": batch_data.page_indexes[0],
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "overall_ocr_res": overall_ocr_res,
                "table_res_list": table_res_list,
                "model_settings": model_settings,
            }
            yield TableRecognitionResult(single_img_res)
