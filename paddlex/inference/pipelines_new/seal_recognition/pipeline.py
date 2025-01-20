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
from .result import SealRecognitionResult
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..doc_preprocessor.result import DocPreprocessorResult

# [TODO] 待更新models_new到models
from ...models_new.object_detection.result import DetResult


class SealRecognitionPipeline(BasePipeline):
    """Seal Recognition Pipeline"""

    entities = ["seal_recognition"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the seal recognition pipeline.

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

        self.use_layout_detection = config.get("use_layout_detection", True)
        if self.use_layout_detection:
            layout_det_config = config.get("SubModules", {}).get(
                "LayoutDetection",
                {"model_config_error": "config error for layout_det_model!"},
            )
            self.layout_det_model = self.create_model(layout_det_config)

        seal_ocr_config = config.get("SubPipelines", {}).get(
            "SealOCR", {"pipeline_config_error": "config error for seal_ocr_pipeline!"}
        )
        self.seal_ocr_pipeline = self.create_pipeline(seal_ocr_config)

        self._crop_by_boxes = CropByBoxes()

        self.batch_sampler = ImageBatchSampler(batch_size=1)

        self.img_reader = ReadImage(format="BGR")

    def check_model_settings_valid(
        self, model_settings: Dict, layout_det_res: DetResult
    ) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_settings (Dict): A dictionary containing input parameters.
            layout_det_res (DetResult): Layout detection result.

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
        return True

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_layout_detection: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_layout_detection (Optional[bool]): Whether to use layout detection.

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
        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_layout_detection=use_layout_detection,
        )

    def predict(
        self,
        input: str | list[str] | np.ndarray | list[np.ndarray],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_layout_detection: Optional[bool] = None,
        layout_det_res: Optional[DetResult] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        **kwargs,
    ) -> SealRecognitionResult:

        model_settings = self.get_model_settings(
            use_doc_orientation_classify, use_doc_unwarping, use_layout_detection
        )

        if not self.check_model_settings_valid(model_settings, layout_det_res):
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

            seal_res_list = []
            seal_region_id = 1
            if not model_settings["use_layout_detection"] and layout_det_res is None:
                layout_det_res = {}
                seal_ocr_res = next(
                    self.seal_ocr_pipeline(
                        doc_preprocessor_image,
                        text_det_limit_side_len=seal_det_limit_side_len,
                        text_det_limit_type=seal_det_limit_type,
                        text_det_thresh=seal_det_thresh,
                        text_det_box_thresh=seal_det_box_thresh,
                        text_det_unclip_ratio=seal_det_unclip_ratio,
                        text_rec_score_thresh=seal_rec_score_thresh,
                    )
                )
                seal_ocr_res["seal_region_id"] = seal_region_id
                seal_res_list.append(seal_ocr_res)
                seal_region_id += 1
            else:
                if model_settings["use_layout_detection"]:
                    layout_det_res = next(self.layout_det_model(doc_preprocessor_image))

                for box_info in layout_det_res["boxes"]:
                    if box_info["label"].lower() in ["seal"]:
                        crop_img_info = self._crop_by_boxes(
                            doc_preprocessor_image, [box_info]
                        )
                        crop_img_info = crop_img_info[0]
                        seal_ocr_res = next(
                            self.seal_ocr_pipeline(
                                crop_img_info["img"],
                                text_det_limit_side_len=seal_det_limit_side_len,
                                text_det_limit_type=seal_det_limit_type,
                                text_det_thresh=seal_det_thresh,
                                text_det_box_thresh=seal_det_box_thresh,
                                text_det_unclip_ratio=seal_det_unclip_ratio,
                                text_rec_score_thresh=seal_rec_score_thresh,
                            )
                        )
                        seal_ocr_res["seal_region_id"] = seal_region_id
                        seal_res_list.append(seal_ocr_res)
                        seal_region_id += 1

            single_img_res = {
                "input_path": input_path,
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "seal_res_list": seal_res_list,
                "model_settings": model_settings,
            }
            yield SealRecognitionResult(single_img_res)
