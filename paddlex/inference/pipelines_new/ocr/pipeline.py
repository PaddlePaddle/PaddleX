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

from typing import Any, Dict, List, Optional
import numpy as np
from scipy.ndimage import rotate
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from ..components import (
    CropByPolys,
    SortQuadBoxes,
    SortPolyBoxes,
    convert_points_to_boxes,
)
from .result import OCRResult
from ..doc_preprocessor.result import DocPreprocessorResult
from ....utils import logging


class OCRPipeline(BasePipeline):
    """OCR Pipeline"""

    entities = "OCR"

    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
    ) -> None:
        """
        Initializes the class with given configurations and options.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """
        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

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
            # TODO: add batch_size
            # batch_size = textline_orientation_config.get("batch_size", 1)
            # self.textline_orientation_model = self.create_model(
            #     textline_orientation_config, batch_size=batch_size
            # )
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
            self.text_det_thresh = text_det_config.get("thresh", 0.3)
            self.text_det_box_thresh = text_det_config.get("box_thresh", 0.6)
            self.text_det_unclip_ratio = text_det_config.get("unclip_ratio", 2.0)
            self._sort_boxes = SortQuadBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="quad")
        elif self.text_type == "seal":
            self.text_det_limit_side_len = text_det_config.get("limit_side_len", 736)
            self.text_det_limit_type = text_det_config.get("limit_type", "min")
            self.text_det_thresh = text_det_config.get("thresh", 0.2)
            self.text_det_box_thresh = text_det_config.get("box_thresh", 0.6)
            self.text_det_unclip_ratio = text_det_config.get("unclip_ratio", 0.5)
            self._sort_boxes = SortPolyBoxes()
            self._crop_by_polys = CropByPolys(det_box_type="poly")
        else:
            raise ValueError("Unsupported text type {}".format(self.text_type))

        self.text_det_model = self.create_model(
            text_det_config,
            limit_side_len=self.text_det_limit_side_len,
            limit_type=self.text_det_limit_type,
            thresh=self.text_det_thresh,
            box_thresh=self.text_det_box_thresh,
            unclip_ratio=self.text_det_unclip_ratio,
        )

        text_rec_config = config.get("SubModules", {}).get(
            "TextRecognition",
            {"model_config_error": "config error for text_rec_model!"},
        )
        # TODO: add batch_size
        # batch_size = text_rec_config.get("batch_size", 1)
        # self.text_rec_model = self.create_model(text_rec_config,
        #     batch_size=batch_size)
        self.text_rec_score_thresh = text_rec_config.get("score_thresh", 0)
        self.text_rec_model = self.create_model(text_rec_config)

        self.batch_sampler = ImageBatchSampler(batch_size=1)
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
            rotated_image = rotate(image_array, rotate_angle, reshape=True)
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
            box_thresh=text_det_box_thresh,
            unclip_ratio=text_det_unclip_ratio,
        )

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
    ) -> OCRResult:
        """
        Predict OCR results based on input images or arrays with optional preprocessing steps.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): Input image of pdf path(s) or numpy array(s).
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            text_det_limit_side_len (Optional[int]): Maximum side length for text detection.
            text_det_limit_type (Optional[str]): Type of limit to apply for text detection.
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
            text_det_thresh,
            text_det_box_thresh,
            text_det_unclip_ratio,
        )

        if text_rec_score_thresh is None:
            text_rec_score_thresh = self.text_rec_score_thresh

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

            det_res = next(
                self.text_det_model(doc_preprocessor_image, **text_det_params)
            )

            dt_polys = det_res["dt_polys"]
            dt_scores = det_res["dt_scores"]

            dt_polys = self._sort_boxes(dt_polys)

            single_img_res = {
                "input_path": input_path,
                "doc_preprocessor_res": doc_preprocessor_res,
                "dt_polys": dt_polys,
                "model_settings": model_settings,
                "text_det_params": text_det_params,
                "text_type": self.text_type,
                "text_rec_score_thresh": text_rec_score_thresh,
            }

            single_img_res["rec_texts"] = []
            single_img_res["rec_scores"] = []
            single_img_res["rec_polys"] = []
            if len(dt_polys) > 0:
                all_subs_of_img = list(
                    self._crop_by_polys(doc_preprocessor_image, dt_polys)
                )
                # use textline orientation model
                if model_settings["use_textline_orientation"]:
                    angles = [
                        textline_angle_info["class_ids"][0]
                        for textline_angle_info in self.textline_orientation_model(
                            all_subs_of_img
                        )
                    ]
                    all_subs_of_img = self.rotate_image(all_subs_of_img, angles)
                else:
                    angles = [-1] * len(all_subs_of_img)
                single_img_res["textline_orientation_angles"] = angles

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
                for idx, rec_res in enumerate(self.text_rec_model(sorted_subs_of_img)):
                    sub_img_id = sorted_subs_info[idx]["sub_img_id"]
                    sub_img_info_list[sub_img_id]["rec_res"] = rec_res
                for sno in range(len(sub_img_info_list)):
                    rec_res = sub_img_info_list[sno]["rec_res"]
                    if rec_res["rec_score"] >= text_rec_score_thresh:
                        single_img_res["rec_texts"].append(rec_res["rec_text"])
                        single_img_res["rec_scores"].append(rec_res["rec_score"])
                        single_img_res["rec_polys"].append(dt_polys[rno])
            if self.text_type == "general":
                rec_boxes = convert_points_to_boxes(single_img_res["rec_polys"])
                single_img_res["rec_boxes"] = rec_boxes
            else:
                single_img_res["rec_boxes"] = np.array([])
            yield OCRResult(single_img_res)
