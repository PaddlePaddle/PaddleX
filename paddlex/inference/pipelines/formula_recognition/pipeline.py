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
from .result import FormulaRecognitionResult


class _FormulaRecognitionPipeline(BasePipeline):
    """Formula Recognition Pipeline"""

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
    ) -> None:
        """Initializes the formula recognition pipeline.

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
            layout_kwargs = {}
            if (threshold := layout_det_config.get("threshold", None)) is not None:
                layout_kwargs["threshold"] = threshold
            if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
                layout_kwargs["layout_nms"] = layout_nms
            if (
                layout_unclip_ratio := layout_det_config.get(
                    "layout_unclip_ratio", None
                )
            ) is not None:
                layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
            if (
                layout_merge_bboxes_mode := layout_det_config.get(
                    "layout_merge_bboxes_mode", None
                )
            ) is not None:
                layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
            self.layout_det_model = self.create_model(
                layout_det_config, **layout_kwargs
            )

        formula_recognition_config = config.get("SubModules", {}).get(
            "FormulaRecognition",
            {"model_config_error": "config error for formula_rec_model!"},
        )
        self.formula_recognition_model = self.create_model(formula_recognition_config)

        self._crop_by_boxes = CropByBoxes()

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")

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

    def check_model_settings_valid(
        self, model_settings: Dict, layout_det_res: Union[DetResult, List[DetResult]]
    ) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            model_settings (Dict): A dictionary containing input parameters.
            layout_det_res (Union[DetResult, List[DetResult]]): The layout detection result(s).
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

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_layout_detection: Optional[bool] = None,
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        layout_det_res: Optional[Union[DetResult, List[DetResult]]] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        **kwargs,
    ) -> FormulaRecognitionResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) of pdf(s) to be processed.
            use_layout_detection (Optional[bool]): Whether to use layout detection.
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            layout_det_res (Optional[Union[DetResult, List[DetResult]]]): The layout detection result(s).
                It will be used if it is not None and use_layout_detection is False.
            **kwargs: Additional keyword arguments.

        Returns:
            formulaRecognitionResult: The predicted formula recognition result.
        """
        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_layout_detection,
        )

        if not self.check_model_settings_valid(model_settings, layout_det_res):
            yield {"error": "the input params for model settings are invalid!"}

        external_layout_det_results = layout_det_res
        if external_layout_det_results is not None:
            if not isinstance(external_layout_det_results, list):
                external_layout_det_results = [external_layout_det_results]
            external_layout_det_results = iter(external_layout_det_results)

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

            formula_results = []

            if (
                not model_settings["use_layout_detection"]
                and external_layout_det_results is None
            ):
                layout_det_results = [{} for _ in doc_preprocessor_images]
                formula_rec_results = list(
                    self.formula_recognition_model(doc_preprocessor_images)
                )
                for formula_rec_res in formula_rec_results:
                    formula_results_for_img = []
                    formula_rec_res["formula_region_id"] = 1
                    formula_results_for_img.append(formula_rec_res)
                    formula_results.append(formula_results_for_img)
            else:
                if model_settings["use_layout_detection"]:
                    layout_det_results = list(
                        self.layout_det_model(
                            doc_preprocessor_images,
                            threshold=layout_threshold,
                            layout_nms=layout_nms,
                            layout_unclip_ratio=layout_unclip_ratio,
                            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                        )
                    )
                else:
                    layout_det_results = []
                    for _ in doc_preprocessor_images:
                        try:
                            layout_det_res = next(external_layout_det_results)
                        except StopIteration:
                            raise ValueError("No more layout det results")
                        layout_det_results.append(layout_det_res)

                formula_crop_imgs = []
                formula_det_results = []
                chunk_indices = [0]
                for doc_preprocessor_image, layout_det_res in zip(
                    doc_preprocessor_images, layout_det_results
                ):
                    formula_region_id = 1
                    for box_info in layout_det_res["boxes"]:
                        if box_info["label"].lower() in ["formula"]:
                            crop_img_info = self._crop_by_boxes(
                                doc_preprocessor_image, [box_info]
                            )
                            crop_img_info = crop_img_info[0]
                            formula_crop_imgs.append(crop_img_info["img"])
                            res = {}
                            res["formula_region_id"] = formula_region_id
                            res["dt_polys"] = box_info["coordinate"]
                            formula_det_results.append(res)
                            formula_region_id += 1
                    chunk_indices.append(len(formula_crop_imgs))

                formula_rec_results = list(
                    self.formula_recognition_model(formula_crop_imgs)
                )
                for idx in range(len(chunk_indices) - 1):
                    formula_det_results_for_idx = formula_det_results[
                        chunk_indices[idx] : chunk_indices[idx + 1]
                    ]
                    formula_rec_results_for_idx = formula_rec_results[
                        chunk_indices[idx] : chunk_indices[idx + 1]
                    ]
                    for formula_det_res, formula_rec_res in zip(
                        formula_det_results_for_idx, formula_rec_results_for_idx
                    ):
                        formula_region_id = formula_det_res["formula_region_id"]
                        dt_polys = formula_det_res["dt_polys"]
                        formula_rec_res["formula_region_id"] = formula_region_id
                        formula_rec_res["dt_polys"] = dt_polys
                    formula_results.append(formula_rec_results_for_idx)

            for (
                input_path,
                page_index,
                layout_det_res,
                doc_preprocessor_res,
                formula_results_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                layout_det_results,
                doc_preprocessor_results,
                formula_results,
            ):
                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "layout_det_res": layout_det_res,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "formula_res_list": formula_results_for_img,
                    "model_settings": model_settings,
                }
                yield FormulaRecognitionResult(single_img_res)


@pipeline_requires_extra("ocr")
class FormulaRecognitionPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = ["formula_recognition"]

    @property
    def _pipeline_cls(self):
        return _FormulaRecognitionPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
