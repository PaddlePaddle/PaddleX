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

import queue
import threading
import time
from itertools import chain
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
from .uilts import convert_otsl_to_html, filter_overlap_boxes, merge_blocks, truncate_repetitive_content, tokenize_figure_of_table, untokenize_figure_of_table

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
        model_name = layout_det_config.get("model_name", None)
        assert (
            model_name is not None and model_name == "PP-DocLayoutV2-L"
        ), "model_name must be PP-DocLayoutV2-L"
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

        self.use_queues = config.get("use_queues", False)

    def close(self):
        self.vl_rec_model.close()

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

    def get_layout_parsing_results(self, images, layout_det_results, imgs_in_doc):
        blocks = []
        block_imgs = []
        text_prompts = []
        is_table_flags = []
        vlm_block_ids = []
        # figure_token_maps = []
        for i, (image, layout_det_res,imgs_in_doc_for_img) in enumerate(zip(images, layout_det_results,imgs_in_doc)):
            layout_det_res = filter_overlap_boxes(layout_det_res)
            boxes = layout_det_res["boxes"]
            blocks_for_img = self.crop_by_boxes(image, boxes)
            blocks_for_img = merge_blocks(blocks_for_img, non_merge_labels=IMAGE_LABELS + ["table"])
            blocks.append(blocks_for_img)
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_label = block["label"]
                if block_label not in IMAGE_LABELS and block_img is not None:
                    text_prompt = "OCR:"
                    if block_label == "table":
                        text_prompt = "Table Recognition:"
                    elif "formula" in block_label:
                        text_prompt = "Formula Recognition:"
                    # figure_token_map = {}
                    # block_img, figure_token_map = tokenize_figure_of_table(block_img, block_bbox, imgs_in_doc_for_img)
                    block_imgs.append(block_img)
                    text_prompts.append(text_prompt)
                    is_table_flags.append(block_label == "table")
                    # figure_token_maps.append(figure_token_map)
                    vlm_block_ids.append((i, j))

        kwargs = {
            "use_cache": True,
            "max_new_tokens": 4096,
        }
        vl_rec_results_table = list(
            self.vl_rec_model.predict(
                [
                    {
                        "image": block_img,
                        "query": text_prompt,
                    }
                    for block_img, text_prompt, is_table in zip(
                        block_imgs, text_prompts, is_table_flags
                    )
                    if is_table
                ],
                skip_special_tokens=False,
                **kwargs,
            )
        )
        vl_rec_results_other = list(
            self.vl_rec_model.predict(
                [
                    {
                        "image": block_img,
                        "query": text_prompt,
                    }
                    for block_img, text_prompt, is_table in zip(
                        block_imgs, text_prompts, is_table_flags
                    )
                    if not is_table
                ],
                skip_special_tokens=True,
                **kwargs,
            )
        )
        vl_rec_results = []
        for is_table in is_table_flags:
            if is_table:
                vl_rec_results.append(vl_rec_results_table.pop(0))
            else:
                vl_rec_results.append(vl_rec_results_other.pop(0))

        parsing_res_lists = []
        vl_rec_res_lists = []
        table_res_lists = []
        curr_vlm_block_idx = 0
        for i, blocks_for_img in enumerate(blocks):
            parsing_res_list = []
            vl_rec_res_list = []
            table_res_list = []
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_bbox = block["box"]
                block_label = block["label"]
                block_content = ""
                if curr_vlm_block_idx < len(vlm_block_ids) and vlm_block_ids[
                    curr_vlm_block_idx
                ] == (i, j):
                    vl_rec_result = vl_rec_results[curr_vlm_block_idx]
                    curr_vlm_block_idx += 1
                    vl_rec_result["image"] = block_img
                    vl_rec_res_list.append(vl_rec_result)
                    result_str = vl_rec_result.get("result", "")
                    if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                    ):
                        result_str = result_str.replace("$", "")

                        result_str = (
                            result_str.replace("\(", " $ ")
                            .replace("\\)", " $ ")
                            .replace("\\[", " $$ ")
                            .replace("\\]", " $$ ")
                        )
                    if block_label == "table":
                        result_str = convert_otsl_to_html(result_str)
                        # figure_token_map = figure_token_maps[curr_vlm_block_idx]
                        # result_str = untokenize_figure_of_table(result_str, figure_token_map)

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
        use_queues: Optional[bool] = None,
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

        if use_queues is None:
            use_queues = self.use_queues

        def _process_cv(batch_data, new_batch_size=None):
            if not new_batch_size:
                new_batch_size = len(batch_data)

            for idx in range(0, len(batch_data), new_batch_size):
                instances = batch_data.instances[idx : idx + new_batch_size]
                input_paths = batch_data.input_paths[idx : idx + new_batch_size]
                page_indexes = batch_data.page_indexes[idx : idx + new_batch_size]

                image_arrays = self.img_reader(instances)

                if model_settings["use_doc_preprocessor"]:
                    doc_preprocessor_results = list(
                        self.doc_preprocessor_pipeline(
                            image_arrays,
                            use_doc_orientation_classify=use_doc_orientation_classify,
                            use_doc_unwarping=use_doc_unwarping,
                        )
                    )
                else:
                    doc_preprocessor_results = [
                        {"output_img": arr} for arr in image_arrays
                    ]

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
                    gather_imgs(doc_pp_img, layout_det_res["boxes"])
                    for doc_pp_img, layout_det_res in zip(
                        doc_preprocessor_images, layout_det_results
                    )
                ]

                yield input_paths, page_indexes, doc_preprocessor_images, doc_preprocessor_results, layout_det_results, imgs_in_doc

        def _process_vlm(results_cv):
            (
                input_paths,
                page_indexes,
                doc_preprocessor_images,
                doc_preprocessor_results,
                layout_det_results,
                imgs_in_doc,
            ) = results_cv

            parsing_res_lists, vl_rec_res_lists, table_res_lists = (
                self.get_layout_parsing_results(
                    doc_preprocessor_images,
                    layout_det_results,
                    imgs_in_doc,
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
                input_paths,
                page_indexes,
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

        if use_queues:
            max_num_batches_in_process = 64
            queue_input = queue.Queue(maxsize=max_num_batches_in_process)
            queue_cv = queue.Queue(maxsize=max_num_batches_in_process)
            queue_vlm = queue.Queue(
                maxsize=self.batch_sampler.batch_size * max_num_batches_in_process
            )
            event_shutdown = threading.Event()
            event_data_loading_done = threading.Event()
            event_cv_processing_done = threading.Event()
            event_vlm_processing_done = threading.Event()

            def _worker_input(input_):
                all_batch_data = self.batch_sampler(input_)
                while not event_shutdown.is_set():
                    try:
                        batch_data = next(all_batch_data)
                    except StopIteration:
                        break
                    except Exception as e:
                        queue_input.put((False, "input", e))
                        break
                    else:
                        queue_input.put((True, batch_data))
                event_data_loading_done.set()

            def _worker_cv():
                while not event_shutdown.is_set():
                    try:
                        item = queue_input.get(timeout=0.5)
                    except queue.Empty:
                        if event_data_loading_done.is_set():
                            event_cv_processing_done.set()
                            break
                        continue
                    if not item[0]:
                        queue_cv.put(item)
                        break
                    try:
                        for results_cv in _process_cv(
                            item[1], self.layout_det_model.batch_sampler.batch_size
                        ):
                            queue_cv.put((True, results_cv))
                    except Exception as e:
                        queue_cv.put((False, "cv", e))
                        break

            def _worker_vlm():
                MAX_QUEUE_DELAY_SECS = 0.5
                MAX_NUM_BOXES = self.vl_rec_model.batch_sampler.batch_size

                while not event_shutdown.is_set():
                    results_cv_list = []
                    start_time = time.time()
                    should_break = False
                    num_boxes = 0
                    while True:
                        remaining_time = MAX_QUEUE_DELAY_SECS - (
                            time.time() - start_time
                        )
                        if remaining_time <= 0:
                            break
                        try:
                            item = queue_cv.get(timeout=remaining_time)
                        except queue.Empty:
                            break
                        if not item[0]:
                            queue_vlm.put(item)
                            should_break = True
                            break
                        results_cv_list.append(item[1])
                        for res in results_cv_list[-1][4]:
                            num_boxes += len(res["boxes"])
                        if num_boxes >= MAX_NUM_BOXES:
                            break
                    if should_break:
                        break
                    if not results_cv_list:
                        if event_cv_processing_done.is_set():
                            event_vlm_processing_done.set()
                            break
                        continue

                    merged_results_cv = [
                        list(chain.from_iterable(lists))
                        for lists in zip(*results_cv_list)
                    ]

                    try:
                        for result_vlm in _process_vlm(merged_results_cv):
                            queue_vlm.put((True, result_vlm))
                    except Exception as e:
                        queue_vlm.put((False, "vlm", e))
                        break

            thread_input = threading.Thread(
                target=_worker_input, args=(input,), daemon=False
            )
            thread_input.start()
            thread_cv = threading.Thread(target=_worker_cv, daemon=False)
            thread_cv.start()
            thread_vlm = threading.Thread(target=_worker_vlm, daemon=False)
            thread_vlm.start()

        try:
            if use_queues:
                while not (event_vlm_processing_done.is_set() and queue_vlm.empty()):
                    try:
                        item = queue_vlm.get(timeout=0.5)
                    except queue.Empty:
                        if event_vlm_processing_done.is_set():
                            break
                        continue
                    if not item[0]:
                        raise RuntimeError(
                            f"Exception from the '{item[1]}' worker: {item[2]}"
                        )
                    else:
                        yield item[1]
            else:
                for batch_data in self.batch_sampler(input):
                    results_cv_list = list(_process_cv(batch_data))
                    assert len(results_cv_list) == 1, len(results_cv_list)
                    results_cv = results_cv_list[0]
                    for res in _process_vlm(results_cv):
                        yield res
        finally:
            if use_queues:
                event_shutdown.set()
                thread_cv.join(timeout=5)
                if thread_cv.is_alive():
                    logging.warning("CV worker did not terminate in time")
                thread_vlm.join(timeout=5)
                if thread_vlm.is_alive():
                    logging.warning("VLM worker did not terminate in time")


@pipeline_requires_extra("ocr")
class PPOCRVLPipeline(AutoParallelImageSimpleInferencePipeline):
    entities = "PP-OCR-VL"

    @property
    def _pipeline_cls(self):
        return _PPOCRVLPipeline

    def _get_batch_size(self, config):
        return config.get("batch_size", 1)
