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

import re
from copy import deepcopy
from time import sleep
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import MarkDownBatchSampler
from ...utils.benchmark import benchmark
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from .result import DocumentResult, LatexResult, MarkdownResult
from .utils import (
    split_original_texts,
    split_text_recursive,
    translate_code_block,
    translate_html_block,
)


@benchmark.time_methods
@pipeline_requires_extra("trans")
class PP_DocTranslation_Pipeline(BasePipeline):
    """
    PP_ DocTranslation_Pipeline
    """

    entities = ["PP-DocTranslation"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        hpi_config: Optional[Union[Dict[str, Any], HPIConfig]] = None,
        initial_predictor: bool = False,
    ) -> None:
        """Initializes the PP_Translation_Pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use the high-performance
                inference plugin (HPIP) by default. Defaults to False.
            hpi_config (Optional[Union[Dict[str, Any], HPIConfig]], optional):
                The default high-performance inference configuration dictionary.
                Defaults to None.
            initial_predictor (bool, optional): Whether to initialize the predictor. Defaults to True.
        """

        super().__init__(
            device=device, pp_option=pp_option, use_hpip=use_hpip, hpi_config=hpi_config
        )

        self.pipeline_name = config["pipeline_name"]
        self.config = config
        self.use_layout_parser = config.get("use_layout_parser", True)

        self.layout_parsing_pipeline = None
        self.chat_bot = None

        if initial_predictor:
            self.inintial_visual_predictor(config)
            self.inintial_chat_predictor(config)

        self.markdown_batch_sampler = MarkDownBatchSampler()

    def close(self):
        if self.layout_parsing_pipeline is not None:
            self.layout_parsing_pipeline.close()

    def inintial_visual_predictor(self, config: dict) -> None:
        """
        Initializes the visual predictor with the given configuration.

        Args:
            config (dict): The configuration dictionary containing the necessary
                                parameters for initializing the predictor.
        Returns:
            None
        """
        self.use_layout_parser = config.get("use_layout_parser", True)

        if self.use_layout_parser:
            layout_parsing_config = config.get("SubPipelines", {}).get(
                "LayoutParser",
                {"pipeline_config_error": "config error for layout_parsing_pipeline!"},
            )
            self.layout_parsing_pipeline = self.create_pipeline(layout_parsing_config)
        return

    def inintial_chat_predictor(self, config: dict) -> None:
        """
        Initializes the chat predictor with the given configuration.

        Args:
            config (dict): The configuration dictionary containing the necessary
                                parameters for initializing the predictor.
        Returns:
            None
        """
        from .. import create_chat_bot

        chat_bot_config = config.get("SubModules", {}).get(
            "LLM_Chat",
            {"chat_bot_config_error": "config error for llm chat bot!"},
        )
        self.chat_bot = create_chat_bot(chat_bot_config)

        from .. import create_prompt_engineering

        translate_pe_config = (
            config.get("SubModules", {})
            .get("PromptEngneering", {})
            .get(
                "Translate_CommonText",
                {"pe_config_error": "config error for translate_pe_config!"},
            )
        )
        self.translate_pe = create_prompt_engineering(translate_pe_config)
        return

    def predict(self, *args, **kwargs) -> None:
        logging.error(
            "PP-Translation Pipeline do not support to call `predict()` directly! Please invoke `visual_predict`, `build_vector`, `chat` sequentially to obtain the result."
        )
        return

    def visual_predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_textline_orientation: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        use_region_detection: Optional[bool] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        use_wired_table_cells_trans_to_html: bool = False,
        use_wireless_table_cells_trans_to_html: bool = False,
        use_table_orientation_classify: bool = True,
        use_ocr_results_with_table_cells: bool = True,
        use_e2e_wired_table_rec_model: bool = False,
        use_e2e_wireless_table_rec_model: bool = True,
        **kwargs,
    ) -> dict:
        """
        This function takes an input image or a list of images and performs various visual
        prediction tasks such as document orientation classification, document unwarping,
        general OCR, seal recognition, and table recognition based on the provided flags.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): Input image path, list of image paths,
                                                                        numpy array of an image, or list of numpy arrays.
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.
            use_formula_recognition (Optional[bool]): Whether to use formula recognition.
            use_region_detection (Optional[bool]): Whether to use region detection.
            layout_threshold (Optional[float]): The threshold value to filter out low-confidence predictions. Default is None.
            layout_nms (bool, optional): Whether to use layout-aware NMS. Defaults to False.
            layout_unclip_ratio (Optional[Union[float, Tuple[float, float]]], optional): The ratio of unclipping the bounding box.
                Defaults to None.
                If it's a single number, then both width and height are used.
                If it's a tuple of two numbers, then they are used separately for width and height respectively.
                If it's None, then no unclipping will be performed.
            layout_merge_bboxes_mode (Optional[str], optional): The mode for merging bounding boxes. Defaults to None.
            text_det_limit_side_len (Optional[int]): Maximum side length for text detection.
            text_det_limit_type (Optional[str]): Type of limit to apply for text detection.
            text_det_thresh (Optional[float]): Threshold for text detection.
            text_det_box_thresh (Optional[float]): Threshold for text detection boxes.
            text_det_unclip_ratio (Optional[float]): Ratio for unclipping text detection boxes.
            text_rec_score_thresh (Optional[float]): Score threshold for text recognition.
            seal_det_limit_side_len (Optional[int]): Maximum side length for seal detection.
            seal_det_limit_type (Optional[str]): Type of limit to apply for seal detection.
            seal_det_thresh (Optional[float]): Threshold for seal detection.
            seal_det_box_thresh (Optional[float]): Threshold for seal detection boxes.
            seal_det_unclip_ratio (Optional[float]): Ratio for unclipping seal detection boxes.
            seal_rec_score_thresh (Optional[float]): Score threshold for seal recognition.
            use_wired_table_cells_trans_to_html (bool): Whether to use wired table cells trans to HTML.
            use_wireless_table_cells_trans_to_html (bool): Whether to use wireless table cells trans to HTML.
            use_table_orientation_classify (bool): Whether to use table orientation classification.
            use_ocr_results_with_table_cells (bool): Whether to use OCR results processed by table cells.
            use_e2e_wired_table_rec_model (bool): Whether to use end-to-end wired table recognition model.
            use_e2e_wireless_table_rec_model (bool): Whether to use end-to-end wireless table recognition model.
            **kwargs (Any): Additional settings to extend functionality.

        Returns:
            dict: A dictionary containing the layout parsing result.
        """
        if self.use_layout_parser == False:
            logging.error("The models for layout parser are not initialized.")
            yield {"error": "The models for layout parser are not initialized."}

        if self.layout_parsing_pipeline is None:
            logging.warning(
                "The layout parsing pipeline is not initialized, will initialize it now."
            )
            self.inintial_visual_predictor(self.config)

        for layout_parsing_result in self.layout_parsing_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
            use_chart_recognition=use_chart_recognition,
            use_region_detection=use_region_detection,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            seal_det_box_thresh=seal_det_box_thresh,
            seal_det_limit_side_len=seal_det_limit_side_len,
            seal_det_limit_type=seal_det_limit_type,
            seal_det_thresh=seal_det_thresh,
            seal_det_unclip_ratio=seal_det_unclip_ratio,
            seal_rec_score_thresh=seal_rec_score_thresh,
            use_wired_table_cells_trans_to_html=use_wired_table_cells_trans_to_html,
            use_wireless_table_cells_trans_to_html=use_wireless_table_cells_trans_to_html,
            use_table_orientation_classify=use_table_orientation_classify,
            use_ocr_results_with_table_cells=use_ocr_results_with_table_cells,
            use_e2e_wired_table_rec_model=use_e2e_wired_table_rec_model,
            use_e2e_wireless_table_rec_model=use_e2e_wireless_table_rec_model,
        ):

            visual_predict_res = {
                "layout_parsing_result": layout_parsing_result,
            }
            yield visual_predict_res

    def load_from_markdown(self, input):

        markdown_info_list = []
        for markdown_sample in self.markdown_batch_sampler.sample(input):
            markdown_content = markdown_sample.instances[0]
            input_path = markdown_sample.input_paths[0]
            markdown_info = {
                "input_path": input_path,
                "page_index": None,
                "markdown_texts": markdown_content,
                "page_continuation_flags": (True, True),
            }
            markdown_info_list.append(MarkdownResult(markdown_info))
        return markdown_info_list

    def chunk_translate(self, md_blocks, chunk_size, translate_func):
        """
        Chunks the given markdown blocks into smaller chunks of size `chunk_size` and translates them using the given
        translate function.

        Args:
            md_blocks (list): A list of tuples representing each block of markdown content. Each tuple consists of a string
          indicating the block type ('text', 'code') and the actual content of the block.
            chunk_size (int): The maximum size of each chunk.
            translate_func (callable): A callable that accepts a string argument and returns the translated version of that string.

        Returns:
            str: A string containing all the translated chunks concatenated together with newlines between them.
        """
        translation_results = []
        chunk = ""
        logging.info(f"Split the original text into {len(md_blocks)} blocks")
        logging.info("Starting translation...")
        for idx, block in enumerate(md_blocks):
            block_type, block_content = block
            if block_type == "code":
                if chunk.strip():
                    translation_results.append(translate_func(chunk.strip()))
                    chunk = ""  # Clear the chunk
                logging.info(f"Translating block {idx+1}/{len(md_blocks)}...")
                translate_code_block(
                    block_content, chunk_size, translate_func, translation_results
                )
            elif len(block_content) < chunk_size and block_type == "text":
                if len(chunk) + len(block_content) < chunk_size:
                    chunk += "\n\n" + block_content
                else:
                    if chunk.strip():
                        logging.info(f"Translating block {idx+1}/{len(md_blocks)}...")
                        translation_results.append(translate_func(chunk.strip()))
                    chunk = block_content
            else:
                logging.info(f"Translating block {idx+1}/{len(md_blocks)}...")
                if chunk.strip():
                    translation_results.append(translate_func(chunk.strip()))
                    chunk = ""  # Clear the chunk

                if block_type == "text":
                    translation_results.append(
                        split_text_recursive(block_content, chunk_size, translate_func)
                    )
                elif block_type == "text_with_html" or block_type == "html":
                    translate_html_block(
                        block_content, chunk_size, translate_func, translation_results
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_type}")

        if chunk.strip():
            translation_results.append(translate_func(chunk.strip()))
        return "\n\n".join(translation_results)

    def translate(
        self,
        ori_md_info_list: List[Dict],
        target_language: str = "zh",
        chunk_size: int = 3000,
        task_description: str = None,
        output_format: str = None,
        rules_str: str = None,
        few_shot_demo_text_content: str = None,
        few_shot_demo_key_value_list: str = None,
        glossary: Dict = None,
        llm_request_interval: float = 0.0,
        chat_bot_config: Dict = None,
        **kwargs,
    ):
        """
        Translate the given original text into the specified target language using the configured translation model.

        Args:
            ori_md_info_list (List[Dict]): A list of dictionaries containing information about the original markdown text to be translated.
            target_language (str, optional): The desired target language code. Defaults to "zh".
            chunk_size (int, optional): The maximum number of characters allowed per chunk when splitting long texts. Defaults to 5000.
            task_description (str, optional): A description of the task being performed by the translation model. Defaults to None.
            output_format (str, optional): The desired output format of the translation result. Defaults to None.
            rules_str (str, optional): Rules or guidelines for the translation model to follow. Defaults to None.
            few_shot_demo_text_content (str, optional): Demo text content for the translation model. Defaults to None.
            few_shot_demo_key_value_list (str, optional): Demo text key-value list for the translation model. Defaults to None.
            glossary (Dict, optional): A dictionary containing terms and their corresponding definitions. Defaults to None.
            llm_request_interval (float, optional): The interval in seconds between each request to the LLM. Defaults to 0.0.
            chat_bot_config (Dict, optional): Configuration for the chat bot used in the translation process. Defaults to None.
            **kwargs: Additional keyword arguments passed to the translation model.

        Yields:
            MarkdownResult: A dictionary containing the translation result in the target language.
        """
        if self.chat_bot is None:
            logging.warning(
                "The LLM chat bot is not initialized,will initialize it now."
            )
            self.inintial_chat_predictor(self.config)

        if chat_bot_config is not None:
            from .. import create_chat_bot

            chat_bot = create_chat_bot(chat_bot_config)
        else:
            chat_bot = self.chat_bot

        if (
            isinstance(ori_md_info_list, list)
            and ori_md_info_list[0].get("page_index") is not None
        ):
            # for multi page pdf
            ori_md_info_list = [self.concatenate_markdown_pages(ori_md_info_list)]

        if not isinstance(llm_request_interval, float):
            llm_request_interval = float(llm_request_interval)

        assert isinstance(glossary, dict) or glossary is None, "glossary must be a dict"

        glossary_str = ""
        if glossary is not None:
            for k, v in glossary.items():
                if isinstance(v, list):
                    v = "或".join(v)
                glossary_str += f"{k}: {v}\n"

        if glossary_str != "":
            if few_shot_demo_key_value_list is None:
                few_shot_demo_key_value_list = glossary_str
            else:
                few_shot_demo_key_value_list += "\n"
                few_shot_demo_key_value_list += glossary_str

        def translate_func(text):
            """
            Translate the given text using the configured translation model.

            Args:
                text (str): The text to be translated.

            Returns:
                str: The translated text in the target language.
            """
            sleep(llm_request_interval)
            prompt = self.translate_pe.generate_prompt(
                original_text=text,
                language=target_language,
                task_description=task_description,
                output_format=output_format,
                rules_str=rules_str,
                few_shot_demo_text_content=few_shot_demo_text_content,
                few_shot_demo_key_value_list=few_shot_demo_key_value_list,
            )
            translate = chat_bot.generate_chat_results(prompt=prompt).get("content", "")

            if "<<END>>" not in translate:
                raise Exception(
                    "The translation did not reach the end. "
                    "This may happen if your chunk_size is too large. Please reduce chunk_size and try again."
                )
            if translate is None:
                raise Exception("The call to the large model failed.")
            translate = translate.replace("<<END>>", "").rstrip()
            return translate

        base_prompt_content = self.translate_pe.generate_prompt(
            original_text="",
            language=target_language,
            task_description=task_description,
            output_format=output_format,
            rules_str=rules_str,
            few_shot_demo_text_content=few_shot_demo_text_content,
            few_shot_demo_key_value_list=few_shot_demo_key_value_list,
        )
        base_prompt_length = len(base_prompt_content)

        if chunk_size > base_prompt_length:
            chunk_size = chunk_size - base_prompt_length
        else:
            raise ValueError(
                f"Chunk size should be greater than the base prompt length ({base_prompt_length}), but got {chunk_size}."
            )

        for ori_md in ori_md_info_list:

            original_texts = ori_md["markdown_texts"]
            md_blocks = split_original_texts(original_texts)
            target_language_texts = self.chunk_translate(
                md_blocks, chunk_size, translate_func
            )

            yield MarkdownResult(
                {
                    "language": target_language,
                    "input_path": ori_md["input_path"],
                    "page_index": ori_md["page_index"],
                    "page_continuation_flags": ori_md["page_continuation_flags"],
                    "markdown_texts": target_language_texts,
                }
            )

    def concatenate_markdown_pages(self, markdown_list: list) -> tuple:
        """
        Concatenate Markdown content from multiple pages into a single document.

        Args:
            markdown_list (list): A list containing Markdown data for each page.

        Returns:
            tuple: A tuple containing the processed Markdown text.
        """
        markdown_texts = ""
        previous_page_last_element_paragraph_end_flag = True

        if len(markdown_list) == 0:
            raise ValueError("The length of markdown_list is zero.")

        for res in markdown_list:
            # Get the paragraph flags for the current page
            page_first_element_paragraph_start_flag: bool = res[
                "page_continuation_flags"
            ][0]
            page_last_element_paragraph_end_flag: bool = res["page_continuation_flags"][
                1
            ]

            # Determine whether to add a space or a newline
            if (
                not page_first_element_paragraph_start_flag
                and not previous_page_last_element_paragraph_end_flag
            ):
                last_char_of_markdown = markdown_texts[-1] if markdown_texts else ""
                first_char_of_handler = (
                    res["markdown_texts"][0] if res["markdown_texts"] else ""
                )

                # Check if the last character and the first character are Chinese characters
                last_is_chinese_char = (
                    re.match(r"[\u4e00-\u9fff]", last_char_of_markdown)
                    if last_char_of_markdown
                    else False
                )
                first_is_chinese_char = (
                    re.match(r"[\u4e00-\u9fff]", first_char_of_handler)
                    if first_char_of_handler
                    else False
                )
                if not (last_is_chinese_char or first_is_chinese_char):
                    markdown_texts += " " + res["markdown_texts"]
                else:
                    markdown_texts += res["markdown_texts"]
            else:
                markdown_texts += "\n\n" + res["markdown_texts"]
            previous_page_last_element_paragraph_end_flag = (
                page_last_element_paragraph_end_flag
            )

        concatenate_result = {
            "input_path": markdown_list[0]["input_path"],
            "page_index": None,
            "page_continuation_flags": (True, True),
            "markdown_texts": markdown_texts,
        }

        return MarkdownResult(concatenate_result)

    def concatenate_word_pages(self, word_list: list) -> tuple:
        """
        Concatenate Word content from multiple pages into a single document.

        Args:
            word_list (list): A list containing Word data for each page.

        Returns:
            tuple: A tuple containing the processed Word document.
        """
        if len(word_list) == 0:
            raise ValueError("The length of word_list is zero.")

        merged_blocks = []
        image = []

        for page_idx, page_blocks in enumerate(word_list):
            for block in page_blocks["word_blocks"]:
                block_copy = deepcopy(block)
                block_copy["page_index"] = page_idx
                merged_blocks.append(block_copy)
            for img_obj in page_blocks["images"]:
                image.append(img_obj)

        return DocumentResult(
            {
                "word_blocks": merged_blocks,
                "input_path": word_list[0]["input_path"],
                "images": image,
            }
        )

    def concatenate_latex_pages(self, latex_info_list: list) -> tuple:
        """
        Concatenate LaTeX content from multiple pages into a single document.

        Args:
            latex_info_list (list): A list containing LaTeX data for each page.

        Returns:
            tuple: A tuple containing the processed LaTeX document.
        """
        if len(latex_info_list) == 0:
            raise ValueError("The length of latex_info_list is zero.")

        merged_blocks = []
        merged_images = []

        for page_idx, page_blocks in enumerate(latex_info_list):
            for block in page_blocks["latex_blocks"]:
                block_copy = deepcopy(block)
                block_copy["page_index"] = page_idx
                merged_blocks.append(block_copy)

            for img_obj in page_blocks["images"]:
                merged_images.append(img_obj)

        return LatexResult(
            {
                "latex_blocks": merged_blocks,
                "images": merged_images,
                "input_path": latex_info_list[0]["input_path"],
            }
        )
