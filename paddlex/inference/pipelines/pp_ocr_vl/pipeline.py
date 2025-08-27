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

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...utils.benchmark import benchmark
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from .._parallel import AutoParallelImageSimpleInferencePipeline
from ..base import BasePipeline
from ..components import CropByBoxes
from ..layout_parsing.utils import gather_imgs
from .result import PPOCRVLBlock, PPOCRVLResult
from .uilts import filter_overlap_boxes, merge_blocks

IMAGE_LABELS = ["image", "header_image", "footer_image", "chart", "seal"]


@benchmark.time_methods
class _PPOCRVLPipeline(BasePipeline):
    """_PPOCRVLPipeline Pipeline"""

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

        layout_det_config = config.get("SubModules", {}).get(
            "LayoutDetection",
            {"model_config_error": "config error for layout_det_model!"},
        )
        # model_name = layout_det_config.get("model_name", None)
        # assert model_name is not None and model_name == "PP-DocLayoutV2-L", "model_name must be PP-DocLayoutV2-L"
        layout_kwargs = {}
        if (threshold := layout_det_config.get("threshold", None)) is not None:
            layout_kwargs["threshold"] = threshold
        if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
            layout_kwargs["layout_nms"] = layout_nms
        if (
            layout_unclip_ratio := layout_det_config.get("layout_unclip_ratio", None)
        ) is not None:
            layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
        if (
            layout_merge_bboxes_mode := layout_det_config.get(
                "layout_merge_bboxes_mode", None
            )
        ) is not None:
            layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
        self.layout_det_model = self.create_model(layout_det_config, **layout_kwargs)

        vl_rec_config = config.get("SubModules", {}).get(
            "VLRecognition",
            {"model_config_error": "config error for vl_rec_model!"},
        )

        self.vl_rec_model = self.create_model(vl_rec_config)

        self.batch_sampler = ImageBatchSampler(batch_size=config.get("batch_size", 1))
        self.img_reader = ReadImage(format="BGR")
        self.crop_by_boxes = CropByBoxes()

    def get_model_settings(
        self,
        use_doc_orientation_classify: Union[bool, None],
        use_doc_unwarping: Union[bool, None],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Union[bool, None]): Enables document orientation classification if True. Defaults to system setting if None.
            use_doc_unwarping (Union[bool, None]): Enables document unwarping if True. Defaults to system setting if None.

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

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
        )

    def check_model_settings_valid(self, input_params: dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized.",
            )
            return False

        return True

    def get_layout_parsing_results(self, images, layout_det_results):
        layout_det_results = [
            filter_overlap_boxes(
                layout_det_res,
            )
            for layout_det_res in layout_det_results
        ]

        blocks = []
        for image, layout_det_res in zip(images, layout_det_results):
            boxes = layout_det_res["boxes"]
            blocks_for_img = self.crop_by_boxes(image, boxes)
            blocks_for_img = merge_blocks(blocks_for_img, non_merge_labels=IMAGE_LABELS)
            blocks.append(blocks_for_img)

        vl_rec_input_flat_list = []
        chunk_indices = [0]
        for blocks_for_img in blocks:
            for block in blocks_for_img:
                block_img = block["img"]
                block_label = block["label"]
                if block_label not in IMAGE_LABELS and block_img is not None:
                    text_prompt = "OCR"
                    if block_label == "table":
                        text_prompt = "Table Recognition:"
                    elif "formula" in block_label:
                        text_prompt = "Formula Recognition:"
                    vl_rec_input_flat_list.append(
                        {
                            "image": block_img,
                            "query": text_prompt,
                        },
                    )
            chunk_indices.append(len(vl_rec_input_flat_list))

        vl_rec_res_flat_list = list(
            self.vl_rec_model.predict(
                vl_rec_input_flat_list,
                use_cache=True,
            )
        )

        parsing_res_lists = []
        vl_rec_res_lists = []
        table_res_lists = []
        for blocks_for_img, idx_st, idx_ed in zip(
            blocks, chunk_indices, chunk_indices[1:]
        ):
            vl_rec_res_list = vl_rec_res_flat_list[idx_st:idx_ed]
            parsing_res_list = []
            table_res_list = []
            for block, vl_rec_result in zip(blocks_for_img, vl_rec_res_list):
                block_bbox = block["box"]
                block_img = block["img"]
                block_label = block["label"]
                vl_rec_result["image"] = block_img
                result_str = vl_rec_result.get("result", "")
                if ("\\(" in result_str and "\\)" in result_str) or (
                    "\\[" in result_str and "\\]" in result_str
                ):
                    result_str = result_str.replace("$", "")

                    result_str = (
                        result_str.replace("\(", " $ ")
                        .replace("\\)", " $ ")
                        .replace("\\[", " ")
                        .replace("\\]", " ")
                    )

                block_content = result_str

                block_info = PPOCRVLBlock(
                    label=block_label,
                    bbox=block_bbox,
                    content=block_content,
                )
                if block_label in IMAGE_LABELS and block_img is not None:
                    x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                    img_path = f"imgs/img_in_{block_label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                    block_info.image = {
                        "path": img_path,
                        "img": Image.fromarray(block_img),
                    }

                parsing_res_list.append(block_info)

            parsing_res_lists.append(parsing_res_list)
            vl_rec_res_lists.append(vl_rec_res_list)
            table_res_lists.append(table_res_list)

        return parsing_res_lists, vl_rec_res_lists, table_res_lists

    def predict(
        self,
        input: Union[str, list[str], np.ndarray, list[np.ndarray]],
        use_doc_orientation_classify: Union[bool, None] = False,
        use_doc_unwarping: Union[bool, None] = False,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        **kwargs,
    ) -> PPOCRVLResult:
        """
        Predicts the layout parsing result for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): Input image path, list of image paths,
                                                                        numpy array of an image, or list of numpy arrays.
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            layout_threshold (Optional[float]): The threshold value to filter out low-confidence predictions. Default is None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[str], optional): The mode for merging bounding boxes. Defaults to None.
            **kwargs (Any): Additional settings to extend functionality.

        Returns:
            PPOCRVLResult: The predicted layout parsing result.
        """
        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
        )

        if not self.check_model_settings_valid(model_settings):
            yield {"error": "the input params for model settings are invalid!"}

        for batch_data in self.batch_sampler(input):
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

            layout_det_results = list(
                self.layout_det_model(
                    doc_preprocessor_images,
                    threshold=layout_threshold,
                    layout_nms=layout_nms,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                )
            )
            imgs_in_doc = [
                gather_imgs(img, res["boxes"])
                for img, res in zip(doc_preprocessor_images, layout_det_results)
            ]

            parsing_res_lists, vl_rec_res_lists, table_res_lists = (
                self.get_layout_parsing_results(
                    doc_preprocessor_images,
                    layout_det_results,
                )
            )

            for (
                input_path,
                page_index,
                doc_preprocessor_image,
                doc_preprocessor_res,
                layout_det_res,
                table_res_list,
                vl_rec_res_list,
                parsing_res_list,
                imgs_in_doc_for_img,
            ) in zip(
                batch_data.input_paths,
                batch_data.page_indexes,
                doc_preprocessor_images,
                doc_preprocessor_results,
                layout_det_results,
                table_res_lists,
                vl_rec_res_lists,
                parsing_res_lists,
                imgs_in_doc,
            ):
                single_img_res = {
                    "input_path": input_path,
                    "page_index": page_index,
                    "doc_preprocessor_res": doc_preprocessor_res,
                    "layout_det_res": layout_det_res,
                    "table_res_list": table_res_list,
                    "vl_rec_res_list": vl_rec_res_list,
                    "parsing_res_list": parsing_res_list,
                    "imgs_in_doc": imgs_in_doc_for_img,
                    "model_settings": model_settings,
                }
                yield PPOCRVLResult(single_img_res)


@pipeline_requires_extra("ocr")
class PPOCRVLPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "PP-OCR-VL"

    @property
    def _pipeline_cls(self):
        return _PPOCRVLPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
