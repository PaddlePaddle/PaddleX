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

from typing import Any, Dict, List, Optional, Union

import numpy as np

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import (
    CropByPolys,
    SortPolyBoxes,
    SortQuadBoxes,
    convert_points_to_boxes,
    rotate_image,
)
from .result import OCRResult


class _OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """
        Initializes the class with given configurations and options.

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

        self.use_textline_orientation = config.get("use_textline_orientation", True)
        if self.use_textline_orientation:
            textline_orientation_config = config.get("SubModules", {}).get(
                "TextLineOrientation",
                {"model_config_error": "config error for textline_orientation_model!"},
            )
            self.textline_orientation_model = self.create_model(
                textline_orientation_config
            )

        text_det_config = config.get("SubModules", {}).get(
            "TextDetection", {"model_config_error": "config error for text_det_model!"}
        )
        self.text_type = config["text_type"]
        if self.text_type == "general":
            self.text_det_limit_side_len = text_det_config.get("limit_side_len", 960)
            self.text_det_limit_type = text_det_config.get("limit_type", "max")
            self.text_det_max_side_limit = text_det_config.get("max_side_limit", 4000)
            self.text_det_thresh = text_det_config.get("thresh", 0.3)
            self.text_det_box_thresh = text_det_config.get("box_thresh", 0.6)
            self.input_shape = text_det_config.get("input_shape", None)
            self.text_det_unclip_ratio = text_det_config.get("unclip_ratio", 2.0)
            self._sort_boxes = SortQuadBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="quad")
        elif self.text_type == "seal":
            self.text_det_limit_side_len = text_det_config.get("limit_side_len", 736)
            self.text_det_limit_type = text_det_config.get("limit_type", "min")
            self.text_det_max_side_limit = text_det_config.get("max_side_limit", 4000)
            self.text_det_thresh = text_det_config.get("thresh", 0.2)
            self.text_det_box_thresh = text_det_config.get("box_thresh", 0.6)
            self.text_det_unclip_ratio = text_det_config.get("unclip_ratio", 0.5)
            self.input_shape = text_det_config.get("input_shape", None)
            self._sort_boxes = SortPolyBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="poly")
        else:
            raise ValueError("Unsupported text type {}".format(self.text_type))

        self.text_det_model = self.create_model(
            text_det_config,
            limit_side_len=self.text_det_limit_side_len,
            limit_type=self.text_det_limit_type,
            max_side_limit=self.text_det_max_side_limit,
            thresh=self.text_det_thresh,
            box_thresh=self.text_det_box_thresh,
            unclip_ratio=self.text_det_unclip_ratio,
            input_shape=self.input_shape,
        )

        text_rec_config = config.get("SubModules", {}).get(
            "TextRecognition",
            {"model_config_error": "config error for text_rec_model!"},
        )
        self.text_rec_score_thresh = text_rec_config.get("score_thresh", 0)
        self.input_shape = text_rec_config.get("input_shape", None)
        self.text_rec_model = self.create_model(
            text_rec_config, input_shape=self.input_shape
        )

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

    def rotate_image(
        self, image_array_list: List[np.ndarray], rotate_angle_list: List[int]
    ) -> List[np.ndarray]:
        """
        Rotate the given image arrays by their corresponding angles.
        0 corresponds to 0 degrees, 1 corresponds to 180 degrees.

        Args:
            image_array_list (List[np.ndarray]): A list of input image arrays to be rotated.
            rotate_angle_list (List[int]): A list of rotation indicators (0 or 1).
                                        0 means rotate by 0 degrees
                                        1 means rotate by 180 degrees

        Returns:
            List[np.ndarray]: A list of rotated image arrays.

        Raises:
            AssertionError: If any rotate_angle is not 0 or 1.
            AssertionError: If the lengths of input lists don't match.
        """
        assert len(image_array_list) == len(
            rotate_angle_list
        ), f"Length of image_array_list ({len(image_array_list)}) must match length of rotate_angle_list ({len(rotate_angle_list)})"

        for angle in rotate_angle_list:
            assert angle in [0, 1], f"rotate_angle must be 0 or 1, now it's {angle}"

        rotated_images = []
        for image_array, rotate_indicator in zip(image_array_list, rotate_angle_list):
            # Convert 0/1 indicator to actual rotation angle
            rotate_angle = rotate_indicator * 180
            rotated_image = rotate_image(image_array, rotate_angle)
            rotated_images.append(rotated_image)

        return rotated_images

    def check_model_settings_valid(self, model_settings: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_info_params(Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if model_settings["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if (
            model_settings["use_textline_orientation"]
            and not self.use_textline_orientation
        ):
            logging.error(
                "Set use_textline_orientation, but the models for use_textline_orientation are not initialized."
            )
            return False

        return True

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_textline_orientation: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation.

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

        if use_textline_orientation is None:
            use_textline_orientation = self.use_textline_orientation
        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_textline_orientation=use_textline_orientation,
        )

    def get_text_det_params(
        self,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_max_side_limit: Optional[int] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
    ) -> dict:
        """
        Get text detection parameters.

        If a parameter is None, its default value from the instance will be used.

        Args:
            text_det_limit_side_len (Optional[int]): The maximum side length of the text box.
            text_det_limit_type (Optional[str]): The type of limit to apply to the text box.
            text_det_max_side_limit (Optional[int]): The maximum side length of the text box.
            text_det_thresh (Optional[float]): The threshold for text detection.
            text_det_box_thresh (Optional[float]): The threshold for the bounding box.
            text_det_unclip_ratio (Optional[float]): The ratio for unclipping the text box.

        Returns:
            dict: A dictionary containing the text detection parameters.
        """
        if text_det_limit_side_len is None:
            text_det_limit_side_len = self.text_det_limit_side_len
        if text_det_limit_type is None:
            text_det_limit_type = self.text_det_limit_type
        if text_det_max_side_limit is None:
            text_det_max_side_limit = self.text_det_max_side_limit
        if text_det_thresh is None:
            text_det_thresh = self.text_det_thresh
        if text_det_box_thresh is None:
            text_det_box_thresh = self.text_det_box_thresh
        if text_det_unclip_ratio is None:
            text_det_unclip_ratio = self.text_det_unclip_ratio
        return dict(
            limit_side_len=text_det_limit_side_len,
            limit_type=text_det_limit_type,
            thresh=text_det_thresh,
            max_side_limit=text_det_max_side_limit,
            box_thresh=text_det_box_thresh,
            unclip_ratio=text_det_unclip_ratio,
        )

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_max_side_limit: Optional[int] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
    ) -> OCRResult:
        """
        Predict OCR results based on input images or arrays with optional preprocessing steps.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): Input image of pdf path(s) or numpy array(s).
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            text_det_limit_side_len (Optional[int]): Maximum side length for text detection.
            text_det_limit_type (Optional[str]): Type of limit to apply for text detection.
            text_det_max_side_limit (Optional[int]): Maximum side length for text detection.
            text_det_thresh (Optional[float]): Threshold for text detection.
            text_det_box_thresh (Optional[float]): Threshold for text detection boxes.
            text_det_unclip_ratio (Optional[float]): Ratio for unclipping text detection boxes.
            text_rec_score_thresh (Optional[float]): Score threshold for text recognition.
        Returns:
            OCRResult: Generator yielding OCR results for each input image.
        """

        model_settings = self.get_model_settings(
            use_doc_orientation_classify, use_doc_unwarping, use_textline_orientation
        )

        if not self.check_model_settings_valid(model_settings):
            yield {"error": "the input params for model settings are invalid!"}

        text_det_params = self.get_text_det_params(
            text_det_limit_side_len,
            text_det_limit_type,
            text_det_max_side_limit,
            text_det_thresh,
            text_det_box_thresh,
            text_det_unclip_ratio,
        )

        if text_rec_score_thresh is None:
            text_rec_score_thresh = self.text_rec_score_thresh

        for _, batch_data in enumerate(self.batch_sampler(input)):
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

            det_results = list(
                self.text_det_model(doc_preprocessor_images, **text_det_params)
            )

            dt_polys_list = [item["dt_polys"] for item in det_results]

            dt_polys_list = [self._sort_boxes(item) for item in dt_polys_list]

            results = [
                {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "dt_polys": dt_polys,
                    "model_settings": model_settings,
                    "text_det_params": text_det_params,
                    "text_type": self.text_type,
                    "text_rec_score_thresh": text_rec_score_thresh,
                    "rec_texts": [],
                    "rec_scores": [],
                    "rec_polys": [],
                }
                for input_path, page_index, doc_preprocessor_res, dt_polys in zip(
                    batch_data.input_paths,
                    batch_data.page_indexes,
                    doc_preprocessor_results,
                    dt_polys_list,
                )
            ]

            indices = list(range(len(doc_preprocessor_images)))
            indices = [idx for idx in indices if len(dt_polys_list[idx]) > 0]

            if indices:
                all_subs_of_imgs = []
                chunk_indices = [0]
                for idx in indices:
                    all_subs_of_img = list(
                        self._crop_by_polys(
                            doc_preprocessor_images[idx], dt_polys_list[idx]
                        )
                    )
                    all_subs_of_imgs.extend(all_subs_of_img)
                    chunk_indices.append(chunk_indices[-1] + len(all_subs_of_img))

                # use textline orientation model
                if model_settings["use_textline_orientation"]:
                    angles = [
                        int(textline_angle_info["class_ids"][0])
                        for textline_angle_info in self.textline_orientation_model(
                            all_subs_of_imgs
                        )
                    ]
                    all_subs_of_imgs = self.rotate_image(all_subs_of_imgs, angles)
                else:
                    angles = [-1] * len(all_subs_of_imgs)
                for i, idx in enumerate(indices):
                    res = results[idx]
                    res["textline_orientation_angles"] = angles[
                        chunk_indices[i] : chunk_indices[i + 1]
                    ]

                # TODO: Process all sub-images in the batch together
                for i, idx in enumerate(indices):
                    all_subs_of_img = all_subs_of_imgs[
                        chunk_indices[i] : chunk_indices[i + 1]
                    ]
                    res = results[idx]
                    dt_polys = dt_polys_list[idx]
                    sub_img_info_list = [
                        {
                            "sub_img_id": img_id,
                            "sub_img_ratio": sub_img.shape[1] / float(sub_img.shape[0]),
                        }
                        for img_id, sub_img in enumerate(all_subs_of_img)
                    ]
                    sorted_subs_info = sorted(
                        sub_img_info_list, key=lambda x: x["sub_img_ratio"]
                    )
                    sorted_subs_of_img = [
                        all_subs_of_img[x["sub_img_id"]] for x in sorted_subs_info
                    ]
                    for i, rec_res in enumerate(
                        self.text_rec_model(sorted_subs_of_img)
                    ):
                        sub_img_id = sorted_subs_info[i]["sub_img_id"]
                        sub_img_info_list[sub_img_id]["rec_res"] = rec_res
                    for sno in range(len(sub_img_info_list)):
                        rec_res = sub_img_info_list[sno]["rec_res"]
                        if rec_res["rec_score"] >= text_rec_score_thresh:
                            res["rec_texts"].append(rec_res["rec_text"])
                            res["rec_scores"].append(rec_res["rec_score"])
                            res["rec_polys"].append(dt_polys[sno])

            for res in results:
                if self.text_type == "general":
                    rec_boxes = convert_points_to_boxes(res["rec_polys"])
                    res["rec_boxes"] = rec_boxes
                else:
                    res["rec_boxes"] = np.array([])

                yield OCRResult(res)


@pipeline_requires_extra("ocr")
class OCRPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "OCR"

    @property
    def _pipeline_cls(self):
        return _OCRPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
