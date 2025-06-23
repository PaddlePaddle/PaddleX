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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ....utils import logging
from ....utils.deps import pipeline_requires_extra
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ...utils.hpi import HPIConfig
from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline
from .result import TranslationMarkdownResult


@pipeline_requires_extra("ie")
class PP_Translation_Pipeline(BasePipeline):
    entities = ["PP-Translation"]

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

        self.batch_sampler = ImageBatchSampler(batch_size=1)
        self.img_reader = ReadImage(format="BGR")

        self.table_structure_len_max = 500

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
        **kwargs,
    ) -> dict:
        """
        This function takes an input image or a list of images and performs various visual
        prediction tasks such as document orientation classification, document unwarping,
        general OCR, seal recognition, and table recognition based on the provided flags.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): Input image path, list of image paths,
                                                                        numpy array of an image, or list of numpy arrays.
            use_doc_orientation_classify (bool): Flag to use document orientation classification.
            use_doc_unwarping (bool): Flag to use document unwarping.
            use_textline_orientation (Optional[bool]): Whether to use textline orientation prediction.
            use_seal_recognition (bool): Flag to use seal recognition.
            use_table_recognition (bool): Flag to use table recognition.
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
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the layout parsing result and visual information.
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
        ):

            visual_predict_res = {
                "layout_parsing_result": layout_parsing_result,
            }
            yield visual_predict_res

    def split_markdown(self, md_text, chunk_size):

        if (
            not isinstance(md_text, str)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError("Invalid input parameters.")

        chunks = []
        current_chunk = []

        # if md_text less than chunk_size, return the md_text
        if len(md_text) < chunk_size:
            chunks.append(md_text)
            return chunks

        # split the md_text into paragraphs
        paragraphs = md_text.split("\n")

        for paragraph in paragraphs:
            if len(paragraph) == 0:
                # 空行直接跳过
                continue

            if len(paragraph) <= chunk_size:
                current_chunk.append(paragraph)
            else:
                # if the paragraph is too long, split it into sentences
                sentences = re.split(r"(?<=[。.!?])", paragraph)
                for sentence in sentences:
                    if len(sentence) == 0:
                        continue

                    if len(sentence) > chunk_size:
                        raise ValueError("A sentence exceeds the chunk size limit.")

                    # if the current chunk is too long, store it and start a new one
                    if sum(len(s) for s in current_chunk) + len(sentence) > chunk_size:
                        chunks.append("\n\n".join(current_chunk))
                        current_chunk = [sentence]
                    else:
                        current_chunk.append(sentence)

            if sum(len(s) for s in current_chunk) >= chunk_size:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def translate(
        self,
        ori_md_info_list: List[Dict],
        target_language: str = "zh",
        chunk_size: int = 5000,
        task_description: str = None,
        output_format: str = None,
        rules_str: str = None,
        few_shot_demo_text_content: str = None,
        few_shot_demo_key_value_list: str = None,
        chat_bot_config=None,
        **kwargs,
    ):
        """
        Translate the given original text into the specified target language using the configured translation model.

        Args:
            original_text (str): The original text to be translated.
            target_language (str): The desired target language code.
            **kwargs: Additional keyword arguments passed to the translation model.

        Returns:
            str: The translated text in the target language.
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

        if len(ori_md_info_list) == 1:
            # for single image or single page pdf
            md_info = ori_md_info_list[0]
        else:
            # for multi page pdf
            md_info = self.concatenate_markdown_pages(ori_md_info_list)

        original_text = md_info["markdown_texts"]

        chunks = self.split_markdown(original_text, chunk_size)

        target_language_md_chunks = []

        if len(chunks) > 1:
            logging.info(
                f"Get the markdown text, it's length is {len(original_text)}, will split it into {len(chunks)} parts."
            )

        logging.info(
            "Starting to translate the markdown text, will take a while. please wait..."
        )
        for idx, chunk in enumerate(chunks):
            logging.info(f"Translating the {idx+1}/{len(chunks)} part.")
            prompt = self.translate_pe.generate_prompt(
                original_text=chunk,
                language=target_language,
                task_description=task_description,
                output_format=output_format,
                rules_str=rules_str,
                few_shot_demo_text_content=few_shot_demo_text_content,
                few_shot_demo_key_value_list=few_shot_demo_key_value_list,
            )
            target_language_md_chunk = chat_bot.generate_chat_results(
                prompt=prompt
            ).get("content", "")

            target_language_md_chunks.append(target_language_md_chunk)

        target_language_md = "\n\n".join(target_language_md_chunks)

        src_result = {
            "language": "src",
            "input_path": md_info["input_path"],
            "page_index": md_info["page_index"],
            "page_continuation_flags": md_info["page_continuation_flags"],
            "markdown_texts": original_text,
        }

        translate_result = {
            "language": target_language,
            "input_path": md_info["input_path"],
            "page_index": md_info["page_index"],
            "page_continuation_flags": md_info["page_continuation_flags"],
            "markdown_texts": target_language_md,
        }
        return TranslationMarkdownResult(src_result), TranslationMarkdownResult(
            translate_result
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
            "page_continuation_flags": False,
            "markdown_texts": markdown_texts,
        }

        return TranslationMarkdownResult(concatenate_result)
