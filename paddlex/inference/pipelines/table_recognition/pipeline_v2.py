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
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils import logging
from ....utils.deps import (
    function_requires_deps,
    is_dep_available,
    pipeline_requires_extra,
)
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...models.object_detection.result import DetResult
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import CropByBoxes
from ..doc_preprocessor.result import DocPreprocessorResult
from ..layout_parsing.utils import get_sub_regions_ocr_res
from ..ocr.result import OCRResult
from .result import SingleTableRecognitionResult, TableRecognitionResult
from .table_recognition_post_processing import (
    get_table_recognition_res as get_table_recognition_res_e2e,
)
from .table_recognition_post_processing_v2 import get_table_recognition_res
from .utils import get_neighbor_boxes_idx

if is_dep_available("scikit-learn"):
    from sklearn.cluster import KMeans


class _TableRecognitionPipelineV2(BasePipeline):
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
        self.general_ocr_pipeline = None
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

        self.table_orientation_classify_model = None
        self.table_orientation_classify_config = config.get("SubModules", {}).get(
            "TableOrientationClassify", None
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

    def cells_det_results_nms(
        self, cells_det_results, cells_det_scores, cells_det_threshold=0.3
    ):
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
            order = order[
                inds + 1
            ]  # inds shifted by 1 because order[0] is the current box
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
        tol = 0
        # Extract coordinates from table_box
        x_min_t, y_min_t, x_max_t, y_max_t = table_box
        adjusted_boxes = []
        for box in ocr_det_boxes:
            x_min_b, y_min_b, x_max_b, y_max_b = box
            # Check if the box is fully inside table_box
            if (
                x_min_b + tol >= x_min_t
                and y_min_b + tol >= y_min_t
                and x_max_b - tol <= x_max_t
                and y_max_b - tol <= y_max_t
            ):
                # Adjust the coordinates to be relative to table_box
                adjusted_box = [
                    x_min_b - x_min_t,  # Adjust x1
                    y_min_b - y_min_t,  # Adjust y1
                    x_max_b - x_min_t,  # Adjust x2
                    y_max_b - y_min_t,  # Adjust y2
                ]
                adjusted_boxes.append(adjusted_box)
            # Discard boxes not fully inside table_box
        return adjusted_boxes

    def cells_det_results_reprocessing(
        self, cells_det_results, cells_det_scores, ocr_det_results, html_pred_boxes_nums
    ):
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
            (box2[2] - box2[0]) * (box2[3] - box2[1])
            # Calculate the IoU
            iou = intersection_area / float(box1_area)
            return iou

        # Function to combine rectangles into N rectangles
        @function_requires_deps("scikit-learn")
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
            centers = np.array(
                [
                    [
                        (rect[0] + rect[2]) / 2,  # Center x-coordinate
                        (rect[1] + rect[3]) / 2,  # Center y-coordinate
                    ]
                    for rect in rectangles
                ]
            )
            # Perform KMeans clustering on the center points to group them into N clusters
            kmeans = KMeans(n_clusters=N, random_state=0, n_init="auto")
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
                if (iou >= iou_threshold) or (sum(merge_ocr_box_iou) >= iou_threshold):
                    has_large_iou = True
                    break
            if not has_large_iou:
                ocr_miss_boxes.append(ocr_rect)
        # If no ocr_miss_boxes, return cells_det_results
        if len(ocr_miss_boxes) == 0:
            final_results = (
                cells_det_results
                if more_cells_flag == True
                else cells_det_results.tolist()
            )
        else:
            if more_cells_flag == True:
                final_results = combine_rectangles(
                    cells_det_results + ocr_miss_boxes, html_pred_boxes_nums
                )
            else:
                # Need to combine ocr_miss_boxes into N rectangles
                N = html_pred_boxes_nums - len(cells_det_results)
                # Combine ocr_miss_boxes into N rectangles
                ocr_supp_boxes = combine_rectangles(ocr_miss_boxes, N)
                # Combine cells_det_results and ocr_supp_boxes
                final_results = np.concatenate(
                    (cells_det_results, ocr_supp_boxes), axis=0
                ).tolist()
        if len(final_results) <= 0.6 * html_pred_boxes_nums:
            final_results = combine_rectangles(ocr_det_results, html_pred_boxes_nums)
        return final_results

    def split_ocr_bboxes_by_table_cells(
        self, cells_det_results, overall_ocr_res, ori_img, k=2
    ):
        """
        Split OCR bounding boxes based on table cell boundaries when they span multiple cells horizontally.

        Args:
            cells_det_results (list): List of cell bounding boxes in format [x1, y1, x2, y2]
            overall_ocr_res (dict): Dictionary containing OCR results with keys:
                                - 'rec_boxes': OCR bounding boxes (will be converted to list)
                                - 'rec_texts': OCR recognized texts
            ori_img (np.array): Original input image array
            k (int): Threshold for determining when to split (minimum number of cells spanned)

        Returns:
            dict: Modified overall_ocr_res with split boxes and texts
        """

        def calculate_iou(box1, box2):
            """
            Calculate Intersection over Union (IoU) between two bounding boxes.

            Args:
                box1 (list): [x1, y1, x2, y2]
                box2 (list): [x1, y1, x2, y2]

            Returns:
                float: IoU value
            """
            # Determine intersection coordinates
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            if x_right < x_left or y_bottom < y_top:
                return 0.0
            # Calculate areas
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            # return intersection_area / float(box1_area + box2_area - intersection_area)
            return intersection_area / box2_area

        def get_overlapping_cells(ocr_box, cells):
            """
            Find cells that overlap significantly with the OCR box (IoU > 0.5).

            Args:
                ocr_box (list): OCR bounding box [x1, y1, x2, y2]
                cells (list): List of cell bounding boxes

            Returns:
                list: Indices of overlapping cells, sorted by x-coordinate
            """
            overlapping = []
            for idx, cell in enumerate(cells):
                if calculate_iou(ocr_box, cell) > 0.5:
                    overlapping.append(idx)
            # Sort overlapping cells by their x-coordinate (left to right)
            overlapping.sort(key=lambda i: cells[i][0])
            return overlapping

        def split_box_by_cells(ocr_box, cell_indices, cells):
            """
            Split OCR box vertically at cell boundaries.

            Args:
                ocr_box (list): Original OCR box [x1, y1, x2, y2]
                cell_indices (list): Indices of cells to split by
                cells (list): All cell bounding boxes

            Returns:
                list: List of split boxes
            """
            if not cell_indices:
                return [ocr_box]
            split_boxes = []
            cells_to_split = [cells[i] for i in cell_indices]
            if ocr_box[0] < cells_to_split[0][0]:
                split_boxes.append(
                    [ocr_box[0], ocr_box[1], cells_to_split[0][0], ocr_box[3]]
                )
            for i in range(len(cells_to_split)):
                current_cell = cells_to_split[i]
                split_boxes.append(
                    [
                        max(ocr_box[0], current_cell[0]),
                        ocr_box[1],
                        min(ocr_box[2], current_cell[2]),
                        ocr_box[3],
                    ]
                )
                if i < len(cells_to_split) - 1:
                    next_cell = cells_to_split[i + 1]
                    if current_cell[2] < next_cell[0]:
                        split_boxes.append(
                            [current_cell[2], ocr_box[1], next_cell[0], ocr_box[3]]
                        )
            last_cell = cells_to_split[-1]
            if last_cell[2] < ocr_box[2]:
                split_boxes.append([last_cell[2], ocr_box[1], ocr_box[2], ocr_box[3]])
            unique_boxes = []
            seen = set()
            for box in split_boxes:
                box_tuple = tuple(box)
                if box_tuple not in seen:
                    seen.add(box_tuple)
                    unique_boxes.append(box)

            return unique_boxes

        # Convert OCR boxes to list if needed
        if hasattr(overall_ocr_res["rec_boxes"], "tolist"):
            ocr_det_results = overall_ocr_res["rec_boxes"].tolist()
        else:
            ocr_det_results = overall_ocr_res["rec_boxes"]
        ocr_texts = overall_ocr_res["rec_texts"]

        # Make copies to modify
        new_boxes = []
        new_texts = []

        # Process each OCR box
        i = 0
        while i < len(ocr_det_results):
            ocr_box = ocr_det_results[i]
            text = ocr_texts[i]
            # Find cells that significantly overlap with this OCR box
            overlapping_cells = get_overlapping_cells(ocr_box, cells_det_results)
            # Check if we need to split (spans >= k cells)
            if len(overlapping_cells) >= k:
                # Split the box at cell boundaries
                split_boxes = split_box_by_cells(
                    ocr_box, overlapping_cells, cells_det_results
                )
                # Process each split box
                split_texts = []
                for box in split_boxes:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    if y2 - y1 > 1 and x2 - x1 > 1:
                        ocr_result = next(
                            self.general_ocr_pipeline.text_rec_model(
                                ori_img[y1:y2, x1:x2, :]
                            )
                        )
                        # Extract the recognized text from the OCR result
                        if "rec_text" in ocr_result:
                            result = ocr_result[
                                "rec_text"
                            ]  # Assumes "rec_texts" contains a single string
                        else:
                            result = ""
                    else:
                        result = ""
                    split_texts.append(result)
                # Add split boxes and texts to results
                new_boxes.extend(split_boxes)
                new_texts.extend(split_texts)
            else:
                # Keep original box and text
                new_boxes.append(ocr_box)
                new_texts.append(text)
            i += 1

        # Update the results dictionary
        overall_ocr_res["rec_boxes"] = new_boxes
        overall_ocr_res["rec_texts"] = new_texts

        return overall_ocr_res

    def gen_ocr_with_table_cells(self, ori_img, cells_bboxes):
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
            if y2 - y1 > 1 and x2 - x1 > 1:
                rec_te = next(self.general_ocr_pipeline(ori_img[y1:y2, x1:x2, :]))
                # Concatenate the texts and append them to the texts_list.
                texts_list.append("".join(rec_te["rec_texts"]))
        # Return the list of recognized texts from each cell.
        return texts_list

    def map_cells_to_original_image(
        self, detections, table_angle, img_width, img_height
    ):
        """
        Map bounding boxes from the rotated image back to the original image.

        Parameters:
        - detections: list of numpy arrays, each containing bounding box coordinates [x1, y1, x2, y2]
        - table_angle: rotation angle in degrees (90, 180, or 270)
        - width_orig: width of the original image (img1)
        - height_orig: height of the original image (img1)

        Returns:
        - mapped_detections: list of numpy arrays with mapped bounding box coordinates
        """

        mapped_detections = []
        for i in range(len(detections)):
            tbx1, tby1, tbx2, tby2 = (
                detections[i][0],
                detections[i][1],
                detections[i][2],
                detections[i][3],
            )
            if table_angle == "270":
                new_x1, new_y1 = tby1, img_width - tbx2
                new_x2, new_y2 = tby2, img_width - tbx1
            elif table_angle == "180":
                new_x1, new_y1 = img_width - tbx2, img_height - tby2
                new_x2, new_y2 = img_width - tbx1, img_height - tby1
            elif table_angle == "90":
                new_x1, new_y1 = img_height - tby2, tbx1
                new_x2, new_y2 = img_height - tby1, tbx2
            new_box = np.array([new_x1, new_y1, new_x2, new_y2])
            mapped_detections.append(new_box)
        return mapped_detections

    def split_string_by_keywords(self, html_string):
        """
        Split HTML string by keywords.

        Args:
            html_string (str): The HTML string.
        Returns:
            split_html (list): The list of html keywords.
        """

        keywords = [
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<tr>",
            "</tr>",
            "<td>",
            "<td",
            ">",
            "</td>",
            'colspan="2"',
            'colspan="3"',
            'colspan="4"',
            'colspan="5"',
            'colspan="6"',
            'colspan="7"',
            'colspan="8"',
            'colspan="9"',
            'colspan="10"',
            'colspan="11"',
            'colspan="12"',
            'colspan="13"',
            'colspan="14"',
            'colspan="15"',
            'colspan="16"',
            'colspan="17"',
            'colspan="18"',
            'colspan="19"',
            'colspan="20"',
            'rowspan="2"',
            'rowspan="3"',
            'rowspan="4"',
            'rowspan="5"',
            'rowspan="6"',
            'rowspan="7"',
            'rowspan="8"',
            'rowspan="9"',
            'rowspan="10"',
            'rowspan="11"',
            'rowspan="12"',
            'rowspan="13"',
            'rowspan="14"',
            'rowspan="15"',
            'rowspan="16"',
            'rowspan="17"',
            'rowspan="18"',
            'rowspan="19"',
            'rowspan="20"',
        ]
        regex_pattern = "|".join(re.escape(keyword) for keyword in keywords)
        split_result = re.split(f"({regex_pattern})", html_string)
        split_html = [part for part in split_result if part]
        return split_html

    def cluster_positions(self, positions, tolerance):
        if not positions:
            return []
        positions = sorted(set(positions))
        clustered = []
        current_cluster = [positions[0]]
        for pos in positions[1:]:
            if abs(pos - current_cluster[-1]) <= tolerance:
                current_cluster.append(pos)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]
        clustered.append(sum(current_cluster) / len(current_cluster))
        return clustered

    def trans_cells_det_results_to_html(self, cells_det_results):
        """
        Trans table cells bboxes to HTML.

        Args:
            cells_det_results (list): The table cells detection results.
        Returns:
            html (list): The list of html keywords.
        """

        tolerance = 5
        x_coords = [x for cell in cells_det_results for x in (cell[0], cell[2])]
        y_coords = [y for cell in cells_det_results for y in (cell[1], cell[3])]
        x_positions = self.cluster_positions(x_coords, tolerance)
        y_positions = self.cluster_positions(y_coords, tolerance)
        x_position_to_index = {x: i for i, x in enumerate(x_positions)}
        y_position_to_index = {y: i for i, y in enumerate(y_positions)}
        num_rows = len(y_positions) - 1
        num_cols = len(x_positions) - 1
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        cells_info = []
        cell_index = 0
        cell_map = {}
        for index, cell in enumerate(cells_det_results):
            x1, y1, x2, y2 = cell
            x1_idx = min(
                range(len(x_positions)), key=lambda i: abs(x_positions[i] - x1)
            )
            x2_idx = min(
                range(len(x_positions)), key=lambda i: abs(x_positions[i] - x2)
            )
            y1_idx = min(
                range(len(y_positions)), key=lambda i: abs(y_positions[i] - y1)
            )
            y2_idx = min(
                range(len(y_positions)), key=lambda i: abs(y_positions[i] - y2)
            )
            col_start = min(x1_idx, x2_idx)
            col_end = max(x1_idx, x2_idx)
            row_start = min(y1_idx, y2_idx)
            row_end = max(y1_idx, y2_idx)
            rowspan = row_end - row_start
            colspan = col_end - col_start
            if rowspan == 0:
                rowspan = 1
            if colspan == 0:
                colspan = 1
            cells_info.append(
                {
                    "row_start": row_start,
                    "col_start": col_start,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "content": "",
                }
            )
            for r in range(row_start, row_start + rowspan):
                for c in range(col_start, col_start + colspan):
                    key = (r, c)
                    if key in cell_map:
                        continue
                    else:
                        cell_map[key] = index
        html = "<table><tbody>"
        for r in range(num_rows):
            html += "<tr>"
            c = 0
            while c < num_cols:
                key = (r, c)
                if key in cell_map:
                    cell_index = cell_map[key]
                    cell_info = cells_info[cell_index]
                    if cell_info["row_start"] == r and cell_info["col_start"] == c:
                        rowspan = cell_info["rowspan"]
                        colspan = cell_info["colspan"]
                        rowspan_attr = f' rowspan="{rowspan}"' if rowspan > 1 else ""
                        colspan_attr = f' colspan="{colspan}"' if colspan > 1 else ""
                        content = cell_info["content"]
                        html += f"<td{rowspan_attr}{colspan_attr}>{content}</td>"
                    c += cell_info["colspan"]
                else:
                    html += "<td></td>"
                    c += 1
            html += "</tr>"
        html += "</tbody></table>"
        html = self.split_string_by_keywords(html)
        return html

    def predict_single_table_recognition_res(
        self,
        image_array: np.ndarray,
        overall_ocr_res: OCRResult,
        table_box: list,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = False,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_ocr_results_with_table_cells: bool = True,
        flag_find_nei_text: bool = True,
    ) -> SingleTableRecognitionResult:
        """
        Predict table recognition results from an image array, layout detection results, and OCR results.

        Args:
            image_array (np.ndarray): The input image represented as a numpy array.
            overall_ocr_res (OCRResult): Overall OCR result obtained after running the OCR pipeline.
                The overall OCR results containing text recognition information.
            table_box (list): The table box coordinates.
            use_e2e_wired_table_rec_model (bool): Whether to use end-to-end wired table recognition model.
            use_e2e_wireless_table_rec_model (bool): Whether to use end-to-end wireless table recognition model.
            use_wired_table_cells_trans_to_html (bool): Whether to use wired table cells trans to HTML.
            use_wireless_table_cells_trans_to_html (bool): Whether to use wireless table cells trans to HTML.
            use_ocr_results_with_table_cells (bool): Whether to use OCR results processed by table cells.
            flag_find_nei_text (bool): Whether to find neighboring text.
        Returns:
            SingleTableRecognitionResult: single table recognition result.
        """

        table_cls_pred = next(self.table_cls_model(image_array))
        table_cls_result = self.extract_results(table_cls_pred, "cls")
        use_e2e_model = False
        cells_trans_to_html = False

        if table_cls_result == "wired_table":
            if use_wired_table_cells_trans_to_html == True:
                cells_trans_to_html = True
            else:
                table_structure_pred = next(self.wired_table_rec_model(image_array))
            if use_e2e_wired_table_rec_model == True:
                use_e2e_model = True
                if cells_trans_to_html == True:
                    table_structure_pred = next(self.wired_table_rec_model(image_array))
            else:
                table_cells_pred = next(
                    self.wired_table_cells_detection_model(image_array, threshold=0.3)
                )  # Setting the threshold to 0.3 can improve the accuracy of table cells detection.
                # If you really want more or fewer table cells detection boxes, the threshold can be adjusted.
        elif table_cls_result == "wireless_table":
            if use_wireless_table_cells_trans_to_html == True:
                cells_trans_to_html = True
            else:
                table_structure_pred = next(self.wireless_table_rec_model(image_array))
            if use_e2e_wireless_table_rec_model == True:
                use_e2e_model = True
                if cells_trans_to_html == True:
                    table_structure_pred = next(
                        self.wireless_table_rec_model(image_array)
                    )
            else:
                table_cells_pred = next(
                    self.wireless_table_cells_detection_model(
                        image_array, threshold=0.3
                    )
                )  # Setting the threshold to 0.3 can improve the accuracy of table cells detection.
                # If you really want more or fewer table cells detection boxes, the threshold can be adjusted.

        if use_e2e_model == False:
            table_cells_result, table_cells_score = self.extract_results(
                table_cells_pred, "det"
            )
            table_cells_result, table_cells_score = self.cells_det_results_nms(
                table_cells_result, table_cells_score
            )
            if cells_trans_to_html == True:
                table_structure_result = self.trans_cells_det_results_to_html(
                    table_cells_result
                )
            else:
                table_structure_result = self.extract_results(
                    table_structure_pred, "table_stru"
                )
                ocr_det_boxes = self.get_region_ocr_det_boxes(
                    overall_ocr_res["rec_boxes"].tolist(), table_box
                )
                table_cells_result = self.cells_det_results_reprocessing(
                    table_cells_result,
                    table_cells_score,
                    ocr_det_boxes,
                    len(table_structure_pred["bbox"]),
                )
            if use_ocr_results_with_table_cells == True:
                if self.cells_split_ocr == True:
                    table_box_copy = np.array([table_box])
                    table_ocr_pred = get_sub_regions_ocr_res(
                        overall_ocr_res, table_box_copy
                    )
                    table_ocr_pred = self.split_ocr_bboxes_by_table_cells(
                        table_cells_result, table_ocr_pred, image_array
                    )
                    cells_texts_list = []
                else:
                    cells_texts_list = self.gen_ocr_with_table_cells(
                        image_array, table_cells_result
                    )
                    table_ocr_pred = {}
            else:
                table_ocr_pred = {}
                cells_texts_list = []
            single_table_recognition_res = get_table_recognition_res(
                table_box,
                table_structure_result,
                table_cells_result,
                overall_ocr_res,
                table_ocr_pred,
                cells_texts_list,
                use_ocr_results_with_table_cells,
                self.cells_split_ocr,
            )
        else:
            cells_texts_list = []
            use_ocr_results_with_table_cells = False
            table_cells_result_e2e = table_structure_pred["bbox"]
            table_cells_result_e2e = [
                [rect[0], rect[1], rect[4], rect[5]] for rect in table_cells_result_e2e
            ]
            if cells_trans_to_html == True:
                table_structure_pred["structure"] = (
                    self.trans_cells_det_results_to_html(table_cells_result_e2e)
                )
            single_table_recognition_res = get_table_recognition_res_e2e(
                table_box,
                table_structure_pred,
                overall_ocr_res,
                cells_texts_list,
                use_ocr_results_with_table_cells,
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
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = False,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
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
            use_e2e_wired_table_rec_model (bool): Whether to use end-to-end wired table recognition model.
            use_e2e_wireless_table_rec_model (bool): Whether to use end-to-end wireless table recognition model.
            use_wired_table_cells_trans_to_html (bool): Whether to use wired table cells trans to HTML.
            use_wireless_table_cells_trans_to_html (bool): Whether to use wireless table cells trans to HTML.
            use_table_orientation_classify (bool): Whether to use table orientation classification.
            use_ocr_results_with_table_cells (bool): Whether to use OCR results processed by table cells.
            **kwargs: Additional keyword arguments.

        Returns:
            TableRecognitionResult: The predicted table recognition result.
        """

        self.cells_split_ocr = True

        if use_table_orientation_classify == True and (
            self.table_orientation_classify_model is None
        ):
            assert self.table_orientation_classify_config != None
            self.table_orientation_classify_model = self.create_model(
                self.table_orientation_classify_config
            )

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
            elif self.general_ocr_pipeline is None and (
                (
                    use_ocr_results_with_table_cells == True
                    and self.cells_split_ocr == False
                )
                or use_table_orientation_classify == True
            ):
                assert self.general_ocr_config_bak != None
                self.general_ocr_pipeline = self.create_pipeline(
                    self.general_ocr_config_bak
                )

            if use_table_orientation_classify == False:
                table_angle = "0"

            table_res_list = []
            table_region_id = 1

            if not model_settings["use_layout_detection"] and layout_det_res is None:
                img_height, img_width = doc_preprocessor_image.shape[:2]
                table_box = [0, 0, img_width - 1, img_height - 1]
                if use_table_orientation_classify == True:
                    table_angle = next(
                        self.table_orientation_classify_model(doc_preprocessor_image)
                    )["label_names"][0]
                if table_angle == "90":
                    doc_preprocessor_image = np.rot90(doc_preprocessor_image, k=1)
                elif table_angle == "180":
                    doc_preprocessor_image = np.rot90(doc_preprocessor_image, k=2)
                elif table_angle == "270":
                    doc_preprocessor_image = np.rot90(doc_preprocessor_image, k=3)
                if table_angle in ["90", "180", "270"]:
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
                    tbx1, tby1, tbx2, tby2 = (
                        table_box[0],
                        table_box[1],
                        table_box[2],
                        table_box[3],
                    )
                    if table_angle == "90":
                        new_x1, new_y1 = tby1, img_width - tbx2
                        new_x2, new_y2 = tby2, img_width - tbx1
                    elif table_angle == "180":
                        new_x1, new_y1 = img_width - tbx2, img_height - tby2
                        new_x2, new_y2 = img_width - tbx1, img_height - tby1
                    elif table_angle == "270":
                        new_x1, new_y1 = img_height - tby2, tbx1
                        new_x2, new_y2 = img_height - tby1, tbx2
                    table_box = [new_x1, new_y1, new_x2, new_y2]
                layout_det_res = {}
                single_table_rec_res = self.predict_single_table_recognition_res(
                    doc_preprocessor_image,
                    overall_ocr_res,
                    table_box,
                    use_e2e_wired_table_rec_model,
                    use_e2e_wireless_table_rec_model,
                    use_wired_table_cells_trans_to_html,
                    use_wireless_table_cells_trans_to_html,
                    use_ocr_results_with_table_cells,
                    flag_find_nei_text=False,
                )
                single_table_rec_res["table_region_id"] = table_region_id
                if use_table_orientation_classify == True and table_angle != "0":
                    img_height, img_width = doc_preprocessor_image.shape[:2]
                    single_table_rec_res["cell_box_list"] = (
                        self.map_cells_to_original_image(
                            single_table_rec_res["cell_box_list"],
                            table_angle,
                            img_width,
                            img_height,
                        )
                    )
                table_res_list.append(single_table_rec_res)
                table_region_id += 1
            else:
                if model_settings["use_layout_detection"]:
                    layout_det_res = next(self.layout_det_model(doc_preprocessor_image))
                img_height, img_width = doc_preprocessor_image.shape[:2]
                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["table"]:
                        crop_img_info = self._crop_by_boxes(
                            doc_preprocessor_image, [box_info]
                        )
                        crop_img_info = crop_img_info[0]
                        table_box = crop_img_info["box"]
                        if use_table_orientation_classify == True:
                            doc_preprocessor_image_copy = doc_preprocessor_image.copy()
                            table_angle = next(
                                self.table_orientation_classify_model(
                                    crop_img_info["img"]
                                )
                            )["label_names"][0]
                        if table_angle == "90":
                            crop_img_info["img"] = np.rot90(crop_img_info["img"], k=1)
                            doc_preprocessor_image_copy = np.rot90(
                                doc_preprocessor_image_copy, k=1
                            )
                        elif table_angle == "180":
                            crop_img_info["img"] = np.rot90(crop_img_info["img"], k=2)
                            doc_preprocessor_image_copy = np.rot90(
                                doc_preprocessor_image_copy, k=2
                            )
                        elif table_angle == "270":
                            crop_img_info["img"] = np.rot90(crop_img_info["img"], k=3)
                            doc_preprocessor_image_copy = np.rot90(
                                doc_preprocessor_image_copy, k=3
                            )
                        if table_angle in ["90", "180", "270"]:
                            overall_ocr_res = next(
                                self.general_ocr_pipeline(
                                    doc_preprocessor_image_copy,
                                    text_det_limit_side_len=text_det_limit_side_len,
                                    text_det_limit_type=text_det_limit_type,
                                    text_det_thresh=text_det_thresh,
                                    text_det_box_thresh=text_det_box_thresh,
                                    text_det_unclip_ratio=text_det_unclip_ratio,
                                    text_rec_score_thresh=text_rec_score_thresh,
                                )
                            )
                            tbx1, tby1, tbx2, tby2 = (
                                table_box[0],
                                table_box[1],
                                table_box[2],
                                table_box[3],
                            )
                            if table_angle == "90":
                                new_x1, new_y1 = tby1, img_width - tbx2
                                new_x2, new_y2 = tby2, img_width - tbx1
                            elif table_angle == "180":
                                new_x1, new_y1 = img_width - tbx2, img_height - tby2
                                new_x2, new_y2 = img_width - tbx1, img_height - tby1
                            elif table_angle == "270":
                                new_x1, new_y1 = img_height - tby2, tbx1
                                new_x2, new_y2 = img_height - tby1, tbx2
                            table_box = [new_x1, new_y1, new_x2, new_y2]
                        single_table_rec_res = (
                            self.predict_single_table_recognition_res(
                                crop_img_info["img"],
                                overall_ocr_res,
                                table_box,
                                use_e2e_wired_table_rec_model,
                                use_e2e_wireless_table_rec_model,
                                use_wired_table_cells_trans_to_html,
                                use_wireless_table_cells_trans_to_html,
                                use_ocr_results_with_table_cells,
                            )
                        )
                        single_table_rec_res["table_region_id"] = table_region_id
                        if (
                            use_table_orientation_classify == True
                            and table_angle != "0"
                        ):
                            img_height_copy, img_width_copy = (
                                doc_preprocessor_image_copy.shape[:2]
                            )
                            single_table_rec_res["cell_box_list"] = (
                                self.map_cells_to_original_image(
                                    single_table_rec_res["cell_box_list"],
                                    table_angle,
                                    img_width_copy,
                                    img_height_copy,
                                )
                            )
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
class TableRecognitionPipelineV2(AutoParallelImageSimpleInferencePipeline):
    entities = ["table_recognition_v2"]

    @property
    def _pipeline_cls(self):
        return _TableRecognitionPipelineV2

    def _get_batch_size(self, config):
        return 1
