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
from ..components import rotate_image
from .result import DocPreprocessorResult


class _DocPreprocessorPipeline(BasePipeline):
    """Doc Preprocessor Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: Optional[str] = None,
        pp_option: Optional[PaddlePredictorOption] = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the doc preprocessor pipeline.

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

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

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
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
    ) -> DocPreprocessorResult:
        """
        Predict the preprocessing result for the input image or images.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) or path(s) to the images or pdfs.
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

        for _, batch_data in enumerate(self.batch_sampler(input)):
            image_arrays = self.img_reader(batch_data.instances)

            if model_settings["use_doc_orientation_classify"]:
                preds = list(self.doc_ori_classify_model(image_arrays))
                angles = []
                rot_imgs = []
                for img, pred in zip(image_arrays, preds):
                    angle = int(pred["label_names"][0])
                    angles.append(angle)
                    rot_img = rotate_image(img, angle)
                    rot_imgs.append(rot_img)
            else:
                angles = [-1 for _ in range(len(image_arrays))]
                rot_imgs = image_arrays

            if model_settings["use_doc_unwarping"]:
                output_imgs = [
                    item["doctr_img"][:, :, ::-1]
                    for item in self.doc_unwarping_model(rot_imgs)
                ]
            else:
                output_imgs = rot_imgs

            for input_path, page_index, image_array, angle, rot_img, output_img in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                image_arrays,
                angles,
                rot_imgs,
                output_imgs,
            ):
                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "input_img": image_array,
                    "model_settings": model_settings,
                    "angle": angle,
                    "rot_img": rot_img,
                    "output_img": output_img,
                }
                yield DocPreprocessorResult(single_img_res)


@pipeline_requires_extra("ocr")
class DocPreprocessorPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "doc_preprocessor"

    @property
    def _pipeline_cls(self):
        return _DocPreprocessorPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
