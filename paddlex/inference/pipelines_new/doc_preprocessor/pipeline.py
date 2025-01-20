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
from scipy.ndimage import rotate
import numpy as np
from ..base import BasePipeline
from .result import DocPreprocessorResult
from ....utils import logging
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ...utils.pp_option import PaddlePredictorOption


class DocPreprocessorPipeline(BasePipeline):
    """Doc Preprocessor Pipeline"""

    entities = "doc_preprocessor"

    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the doc preprocessor pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        self.use_doc_orientation_classify = config.get(
            "use_doc_orientation_classify", True
        )
        if self.use_doc_orientation_classify:
            doc_ori_classify_config = config.get("SubModules", {}).get(
                "DocOrientationClassify",
                {"model_config_error": "config error for doc_ori_classify_model!"},
            )
            self.doc_ori_classify_model = self.create_model(doc_ori_classify_config)

        self.use_doc_unwarping = config.get("use_doc_unwarping", True)
        if self.use_doc_unwarping:
            doc_unwarping_config = config.get("SubModules", {}).get(
                "DocUnwarping",
                {"model_config_error": "config error for doc_unwarping_model!"},
            )
            self.doc_unwarping_model = self.create_model(doc_unwarping_config)

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

    def rotate_image(self, image_array: np.ndarray, rotate_angle: float) -> np.ndarray:
        """
        Rotate the given image array by the specified angle.

        Args:
            image_array (np.ndarray): The input image array to be rotated.
            rotate_angle (float): The angle in degrees by which to rotate the image.

        Returns:
            np.ndarray: The rotated image array.

        Raises:
            AssertionError: If rotate_angle is not in the range [0, 360).
        """
        assert (
            rotate_angle >= 0 and rotate_angle < 360
        ), "rotate_angle must in [0-360), but get {rotate_angle}."
        return rotate(image_array, rotate_angle, reshape=True)

    def check_model_settings_valid(self, model_settings: Dict) -> bool:
        """
        Check if the the input params for model settings are valid based on the initialized models.

        Args:
            model_settings (Dict): A dictionary containing model settings.

        Returns:
            bool: True if all required models are initialized according to the model settings, False otherwise.
        """

        if (
            model_settings["use_doc_orientation_classify"]
            and not self.use_doc_orientation_classify
        ):
            logging.error(
                "Set use_doc_orientation_classify, but the model for doc orientation classify is not initialized."
            )
            return False

        if model_settings["use_doc_unwarping"] and not self.use_doc_unwarping:
            logging.error(
                "Set use_doc_unwarping, but the model for doc unwarping is not initialized."
            )
            return False

        return True

    def get_model_settings(
        self, use_doc_orientation_classify, use_doc_unwarping
    ) -> dict:
        """
        Retrieve the model settings dictionary based on input parameters.

        Args:
            use_doc_orientation_classify (bool, optional): Whether to use document orientation classification.
            use_doc_unwarping (bool, optional): Whether to use document unwarping.

        Returns:
            dict: A dictionary containing the model settings.
        """
        if use_doc_orientation_classify is None:
            use_doc_orientation_classify = self.use_doc_orientation_classify
        if use_doc_unwarping is None:
            use_doc_unwarping = self.use_doc_unwarping
        model_settings = {
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
        }
        return model_settings

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
    ) -> DocPreprocessorResult:
        """
        Predict the preprocessing result for the input image or images.

        Args:
            input (str | list[str] | np.ndarray | list[np.ndarray]): The input image(s) or path(s) to the images or pdfs.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            **kwargs: Additional keyword arguments.

        Returns:
            DocPreprocessorResult: A generator yielding preprocessing results.
        """

        model_settings = self.get_model_settings(
            use_doc_orientation_classify, use_doc_unwarping
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

            if model_settings["use_doc_orientation_classify"]:
                pred = next(self.doc_ori_classify_model(image_array))
                angle = int(pred["label_names"][0])
                rot_img = self.rotate_image(image_array, angle)
            else:
                angle = -1
                rot_img = image_array

            if model_settings["use_doc_unwarping"]:
                output_img = next(self.doc_unwarping_model(rot_img))["doctr_img"]
            else:
                output_img = rot_img

            single_img_res = {
                "input_path": input_path,
                "input_img": image_array,
                "model_settings": model_settings,
                "angle": angle,
                "rot_img": rot_img,
                "output_img": output_img,
            }
            yield DocPreprocessorResult(single_img_res)
