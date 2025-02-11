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

from typing import Any, Dict, Optional, Union, List, Tuple
import re
import cv2
import json
import base64
import numpy as np
import copy
from .pipeline_base import PP_ChatOCR_Pipeline
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ..layout_parsing.result import LayoutParsingResult
from ..components.chat_server import BaseChat


class PP_ChatOCRv4_Pipeline(PP_ChatOCR_Pipeline):
    """PP-ChatOCRv4 Pipeline"""

    entities = ["PP-ChatOCRv4-doc"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
        initial_predictor: bool = True,
    ) -> None:
        """Initializes the pp-chatocrv3-doc pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
            use_layout_parsing (bool, optional): Whether to use layout parsing. Defaults to True.
            initial_predictor (bool, optional): Whether to initialize the predictor. Defaults to True.
        """

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        self.pipeline_name = config["pipeline_name"]
        self.config = config
        self.use_layout_parser = config.get("use_layout_parser", True)
        self.use_mllm_predict = config.get("use_mllm_predict", True)

        self.layout_parsing_pipeline = None
        self.chat_bot = None
        self.retriever = None
        self.mllm_chat_bot = None

        if initial_predictor:
            self.inintial_visual_predictor(config)
            self.inintial_chat_predictor(config)
            self.inintial_mllm_predictor(config)

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

    def inintial_retriever_predictor(self, config: dict) -> None:
        """
        Initializes the retriever predictor with the given configuration.

        Args:
            config (dict): The configuration dictionary containing the necessary
                                parameters for initializing the predictor.
        Returns:
            None
        """
        from .. import create_retriever

        retriever_config = config.get("SubModules", {}).get(
            "LLM_Retriever",
            {"retriever_config_error": "config error for llm retriever!"},
        )
        self.retriever = create_retriever(retriever_config)

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

        text_pe_config = (
            config.get("SubModules", {})
            .get("PromptEngneering", {})
            .get(
                "KIE_CommonText",
                {"pe_config_error": "config error for text_pe!"},
            )
        )
        self.text_pe = create_prompt_engineering(text_pe_config)

        table_pe_config = (
            config.get("SubModules", {})
            .get("PromptEngneering", {})
            .get(
                "KIE_Table",
                {"pe_config_error": "config error for table_pe!"},
            )
        )
        self.table_pe = create_prompt_engineering(table_pe_config)
        return

    def inintial_mllm_predictor(self, config: dict) -> None:
        """
        Initializes the predictor with the given configuration.

        Args:
            config (dict): The configuration dictionary containing the necessary
                                parameters for initializing the predictor.
        Returns:
            None
        """
        from .. import create_chat_bot, create_prompt_engineering

        self.use_mllm_predict = config.get("use_mllm_predict", True)
        if self.use_mllm_predict:
            mllm_chat_bot_config = config.get("SubModules", {}).get(
                "MLLM_Chat",
                {"mllm_chat_bot_config": "config error for mllm chat bot!"},
            )
            self.mllm_chat_bot = create_chat_bot(mllm_chat_bot_config)
            ensemble_pe_config = (
                config.get("SubModules", {})
                .get("PromptEngneering", {})
                .get(
                    "Ensemble",
                    {"pe_config_error": "config error for ensemble_pe!"},
                )
            )
            self.ensemble_pe = create_prompt_engineering(ensemble_pe_config)
        return

    def decode_visual_result(self, layout_parsing_result: LayoutParsingResult) -> dict:
        """
        Decodes the visual result from the layout parsing result.

        Args:
            layout_parsing_result (LayoutParsingResult): The result of layout parsing.

        Returns:
            dict: The decoded visual information.
        """
        text_paragraphs_ocr_res = layout_parsing_result["text_paragraphs_ocr_res"]
        seal_res_list = layout_parsing_result["seal_res_list"]
        normal_text_dict = {}

        for seal_res in seal_res_list:
            for text in seal_res["rec_texts"]:
                layout_type = "印章"
                if layout_type not in normal_text_dict:
                    normal_text_dict[layout_type] = f"{text}"
                else:
                    normal_text_dict[layout_type] += f"\n {text}"

        for text in text_paragraphs_ocr_res["rec_texts"]:
            layout_type = "words in text block"
            if layout_type not in normal_text_dict:
                normal_text_dict[layout_type] = text
            else:
                normal_text_dict[layout_type] += f"\n {text}"

        table_res_list = layout_parsing_result["table_res_list"]
        table_text_list = []
        table_html_list = []
        table_nei_text_list = []
        for table_res in table_res_list:
            table_html_list.append(table_res["pred_html"])
            single_table_text = " ".join(table_res["table_ocr_pred"]["rec_texts"])
            table_text_list.append(single_table_text)
            table_nei_text_list.append(table_res["neighbor_texts"])

        visual_info = {}
        visual_info["normal_text_dict"] = normal_text_dict
        visual_info["table_text_list"] = table_text_list
        visual_info["table_html_list"] = table_html_list
        visual_info["table_nei_text_list"] = table_nei_text_list
        return visual_info

    # Function to perform visual prediction on input images
    def visual_predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_general_ocr: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
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
            use_general_ocr (bool): Flag to use general OCR.
            use_seal_recognition (bool): Flag to use seal recognition.
            use_table_recognition (bool): Flag to use table recognition.
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
            use_general_ocr=use_general_ocr,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
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

            visual_info = self.decode_visual_result(layout_parsing_result)

            visual_predict_res = {
                "layout_parsing_result": layout_parsing_result,
                "visual_info": visual_info,
            }
            yield visual_predict_res

    def save_visual_info_list(self, visual_info: dict, save_path: str) -> None:
        """
        Save the visual info list to the specified file path.

        Args:
            visual_info (dict): The visual info result, which can be a single object or a list of objects.
            save_path (str): The file path to save the visual info list.

        Returns:
            None
        """
        if not isinstance(visual_info, list):
            visual_info_list = [visual_info]
        else:
            visual_info_list = visual_info

        with open(save_path, "w") as fout:
            fout.write(json.dumps(visual_info_list, ensure_ascii=False) + "\n")
        return

    def load_visual_info_list(self, data_path: str) -> List[dict]:
        """
        Loads visual info list from a JSON file.

        Args:
            data_path (str): The path to the JSON file containing visual info.

        Returns:
            list[dict]: A list of dict objects parsed from the JSON file.
        """
        with open(data_path, "r") as fin:
            data = fin.readline()
            visual_info_list = json.loads(data)
        return visual_info_list

    def merge_visual_info_list(
        self, visual_info_list: List[dict]
    ) -> Tuple[list, list, list, list]:
        """
        Merge visual info lists.

        Args:
            visual_info_list (list[dict]): A list of visual info results.

        Returns:
            tuple[list, list, list, list]: A tuple containing four lists, one for normal text dicts,
                                               one for table text lists, one for table HTML lists.
                                               one for table neighbor texts.
        """
        all_normal_text_list = []
        all_table_text_list = []
        all_table_html_list = []
        all_table_nei_text_list = []
        for single_visual_info in visual_info_list:
            normal_text_dict = single_visual_info["normal_text_dict"]
            for key in normal_text_dict:
                normal_text_dict[key] = normal_text_dict[key].replace("\n", "")
            table_text_list = single_visual_info["table_text_list"]
            table_html_list = single_visual_info["table_html_list"]
            table_nei_text_list = single_visual_info["table_nei_text_list"]
            all_normal_text_list.append(normal_text_dict)
            all_table_text_list.extend(table_text_list)
            all_table_html_list.extend(table_html_list)
            all_table_nei_text_list.extend(table_nei_text_list)
        return (
            all_normal_text_list,
            all_table_text_list,
            all_table_html_list,
            all_table_nei_text_list,
        )

    def build_vector(
        self,
        visual_info: dict,
        min_characters: int = 3500,
        llm_request_interval: float = 1.0,
        flag_save_bytes_vector: bool = False,
        retriever_config: dict = None,
    ) -> dict:
        """
        Build a vector representation from visual information.

        Args:
            visual_info (dict): The visual information input, can be a single instance or a list of instances.
            min_characters (int): The minimum number of characters required for text processing, defaults to 3500.
            llm_request_interval (float): The interval between LLM requests, defaults to 1.0.
            flag_save_bytes_vector (bool): Whether to save the vector as bytes, defaults to False.
            retriever_config (dict): The configuration for the retriever, defaults to None.

        Returns:
            dict: A dictionary containing the vector info and a flag indicating if the text is too short.
        """

        if not isinstance(visual_info, list):
            visual_info_list = [visual_info]
        else:
            visual_info_list = visual_info

        if retriever_config is not None:
            from .. import create_retriever

            retriever = create_retriever(retriever_config)
        else:
            if self.retriever is None:
                logging.warning(
                    "The retriever is not initialized,will initialize it now."
                )
                self.inintial_retriever_predictor(self.config)
            retriever = self.retriever

        all_visual_info = self.merge_visual_info_list(visual_info_list)
        (
            all_normal_text_list,
            all_table_text_list,
            all_table_html_list,
            all_table_nei_text_list,
        ) = all_visual_info

        vector_info = {}

        all_items = []
        for i, normal_text_dict in enumerate(all_normal_text_list):
            for type, text in normal_text_dict.items():
                all_items += [f"{type}：{text}\n"]

        for table_html, table_text, table_nei_text in zip(
            all_table_html_list, all_table_text_list, all_table_nei_text_list
        ):
            if len(table_html) > min_characters - self.table_structure_len_max:
                all_items += [f"table：{table_text}\t{table_nei_text}"]

        all_text_str = "".join(all_items)
        vector_info["flag_save_bytes_vector"] = False
        if len(all_text_str) > min_characters:
            vector_info["flag_too_short_text"] = False
            vector_info["vector"] = retriever.generate_vector_database(all_items)
            if flag_save_bytes_vector:
                vector_info["vector"] = retriever.encode_vector_store_to_bytes(
                    vector_info["vector"]
                )
                vector_info["flag_save_bytes_vector"] = True
        else:
            vector_info["flag_too_short_text"] = True
            vector_info["vector"] = all_items
        return vector_info

    def save_vector(self, vector_info: dict, save_path: str) -> None:
        with open(save_path, "w") as fout:
            fout.write(json.dumps(vector_info, ensure_ascii=False) + "\n")
        return

    def load_vector(self, data_path: str) -> dict:
        vector_info = None
        if self.retriever is None:
            logging.warning("The retriever is not initialized,will initialize it now.")
            self.inintial_retriever_predictor(self.config)

        with open(data_path, "r") as fin:
            data = fin.readline()
            vector_info = json.loads(data)
            if (
                "flag_too_short_text" not in vector_info
                or "flag_save_bytes_vector" not in vector_info
                or "vector" not in vector_info
            ):
                logging.error("Invalid vector info.")
                return {"error": "Invalid vector info when load vector!"}

            if vector_info["flag_save_bytes_vector"]:
                vector_info["vector"] = self.retriever.decode_vector_store_from_bytes(
                    vector_info["vector"]
                )
        return vector_info

    def format_key(self, key_list: Union[str, List[str]]) -> List[str]:
        """
        Formats the key list.

        Args:
            key_list (str|list[str]): A string or a list of strings representing the keys.

        Returns:
            list[str]: A list of formatted keys.
        """
        if key_list == "":
            return []

        if isinstance(key_list, list):
            key_list = [key.replace("\xa0", " ") for key in key_list]
            return key_list

        if isinstance(key_list, str):
            key_list = re.sub(r"[\t\n\r\f\v]", "", key_list)
            key_list = key_list.replace("，", ",").split(",")
            return key_list

        return []

    def mllm_pred(
        self,
        input: Union[str, np.ndarray],
        key_list,
        **kwargs,
    ) -> dict:
        if self.use_mllm_predict == False:
            logging.error("MLLM prediction is disabled.")
            return {"mllm_res": "Error:MLLM prediction is disabled!"}

        key_list = self.format_key(key_list)
        if len(key_list) == 0:
            return {"mllm_res": "Error:输入的key_list无效！"}

        if isinstance(input, list):
            logging.error("Input is a list, but it's not supported here.")
            return {"mllm_res": "Error:Input is a list, but it's not supported here!"}
        image_array_list = self.img_reader([input])
        if (
            isinstance(input, str)
            and input.endswith(".pdf")
            and len(image_array_list) > 1
        ):
            logging.error("The input with PDF should have only one page.")
            return {"mllm_res": "Error:The input with PDF should have only one page!"}

        if self.mllm_chat_bot is None:
            logging.warning(
                "The MLLM chat bot is not initialized,will initialize it now."
            )
            self.inintial_mllm_predictor(self.config)

        for image_array in image_array_list:

            assert len(image_array.shape) == 3
            image_string = cv2.imencode(".jpg", image_array)[1].tostring()
            image_base64 = base64.b64encode(image_string).decode("utf-8")
            result = {}
            for key in key_list:
                prompt = (
                    str(key)
                    + "\n请用图片中完整出现的内容回答，可以是单词、短语或句子，针对问题回答尽可能详细和完整，并保持格式、单位、符号和标点都与图片中的文字内容完全一致。"
                )
                mllm_chat_bot_result = self.mllm_chat_bot.generate_chat_results(
                    prompt=prompt, image=image_base64
                )
                if mllm_chat_bot_result is None:
                    return {"mllm_res": "大模型调用失败"}
                result[key] = mllm_chat_bot_result
            return {"mllm_res": result}

    def generate_and_merge_chat_results(
        self,
        chat_bot: BaseChat,
        prompt: str,
        key_list: list,
        final_results: dict,
        failed_results: list,
    ) -> None:
        """
        Generate and merge chat results into the final results dictionary.

        Args:
            prompt (str): The input prompt for the chat bot.
            key_list (list): A list of keys to track which results to merge.
            final_results (dict): The dictionary to store the final merged results.
            failed_results (list): A list of failed results to avoid merging.

        Returns:
            None
        """

        llm_result = chat_bot.generate_chat_results(prompt)
        if llm_result is None:
            logging.error(
                "chat bot error: \n [prompt:]\n %s\n [result:] %s\n"
                % (prompt, self.chat_bot.ERROR_MASSAGE)
            )
            return

        llm_result = self.chat_bot.fix_llm_result_format(llm_result)

        for key, value in llm_result.items():
            if value not in failed_results and key in key_list:
                key_list.remove(key)
                final_results[key] = value
        return

    def get_related_normal_text(
        self,
        retriever_config: dict,
        use_vector_retrieval: bool,
        vector_info: dict,
        key_list: List[str],
        all_normal_text_list: list,
        min_characters: int,
    ) -> str:
        """
        Retrieve related normal text based on vector retrieval or all normal text list.

        Args:
            retriever_config (dict): Configuration for the retriever.
            use_vector_retrieval (bool): Whether to use vector retrieval.
            vector_info (dict): Dictionary containing vector information.
            key_list (list[str]): List of keys to generate question keys.
            all_normal_text_list (list): List of normal text.
            min_characters (int): The minimum number of characters required for text processing, defaults to 3500.

        Returns:
            str: Related normal text.
        """

        if use_vector_retrieval and vector_info is not None:

            if retriever_config is not None:
                from .. import create_retriever

                retriever = create_retriever(retriever_config)
            else:
                if self.retriever is None:
                    logging.warning(
                        "The retriever is not initialized,will initialize it now."
                    )
                    self.inintial_retriever_predictor(self.config)
                retriever = self.retriever

            question_key_list = [f"{key}" for key in key_list]
            vector = vector_info["vector"]
            if not vector_info["flag_too_short_text"]:
                related_text = retriever.similarity_retrieval(
                    question_key_list, vector, topk=50, min_characters=min_characters
                )
            else:
                if len(vector) > 0:
                    related_text = "".join(vector)
                else:
                    related_text = ""
        else:
            all_items = []
            for i, normal_text_dict in enumerate(all_normal_text_list):
                for type, text in normal_text_dict.items():
                    all_items += [f"{type}：{text}\n"]
            related_text = "".join(all_items)
            if len(related_text) > min_characters:
                logging.warning(
                    "The input text content is too long, the large language model may truncate it."
                )
        return related_text

    def ensemble_ocr_llm_mllm(
        self, key_list: List[str], ocr_llm_predict_dict: dict, mllm_predict_dict: dict
    ) -> dict:
        """
        Ensemble OCR_LLM and LMM predictions based on given key list.

        Args:
            key_list (list[str]): List of keys to retrieve predictions.
            ocr_llm_predict_dict (dict): Dictionary containing OCR LLM predictions.
            mllm_predict_dict (dict): Dictionary containing mLLM predictions.

        Returns:
            dict: A dictionary with final predictions.
        """
        final_predict_dict = {}

        for key in key_list:
            predict = ""
            ocr_llm_predict = ""
            mllm_predict = ""
            if key in ocr_llm_predict_dict:
                ocr_llm_predict = ocr_llm_predict_dict[key]
            if key in mllm_predict_dict:
                mllm_predict = mllm_predict_dict[key]
            if ocr_llm_predict != "" and mllm_predict != "":
                prompt = self.ensemble_pe.generate_prompt(
                    key, ocr_llm_predict, mllm_predict
                )
                llm_result = self.chat_bot.generate_chat_results(prompt)
                if llm_result is not None:
                    llm_result = self.chat_bot.fix_llm_result_format(llm_result)
                if key in llm_result:
                    tmp = llm_result[key]
                    if "B" in tmp:
                        predict = mllm_predict
                    else:
                        predict = ocr_llm_predict
                else:
                    predict = ocr_llm_predict
            elif key in ocr_llm_predict_dict:
                predict = ocr_llm_predict_dict[key]
            elif key in mllm_predict_dict:
                predict = mllm_predict_dict[key]

            if predict != "":
                final_predict_dict[key] = predict
        return final_predict_dict

    def chat(
        self,
        key_list: Union[str, List[str]],
        visual_info: dict,
        use_vector_retrieval: bool = True,
        vector_info: dict = None,
        min_characters: int = 3500,
        text_task_description: str = None,
        text_output_format: str = None,
        text_rules_str: str = None,
        text_few_shot_demo_text_content: str = None,
        text_few_shot_demo_key_value_list: str = None,
        table_task_description: str = None,
        table_output_format: str = None,
        table_rules_str: str = None,
        table_few_shot_demo_text_content: str = None,
        table_few_shot_demo_key_value_list: str = None,
        mllm_predict_info: dict = None,
        mllm_integration_strategy: str = "integration",
        chat_bot_config: dict = None,
        retriever_config: dict = None,
    ) -> dict:
        """
        Generates chat results based on the provided key list and visual information.

        Args:
            key_list (Union[str, list[str]]): A single key or a list of keys to extract information.
            visual_info (dict): The visual information result.
            use_vector_retrieval (bool): Whether to use vector retrieval.
            vector_info (dict): The vector information for retrieval.
            min_characters (int): The minimum number of characters required for text processing, defaults to 3500.
            text_task_description (str): The description of the text task.
            text_output_format (str): The output format for text results.
            text_rules_str (str): The rules for generating text results.
            text_few_shot_demo_text_content (str): The text content for few-shot demos.
            text_few_shot_demo_key_value_list (str): The key-value list for few-shot demos.
            table_task_description (str): The description of the table task.
            table_output_format (str): The output format for table results.
            table_rules_str (str): The rules for generating table results.
            table_few_shot_demo_text_content (str): The text content for table few-shot demos.
            table_few_shot_demo_key_value_list (str): The key-value list for table few-shot demos.
            mllm_predict_dict (dict): The dictionary of mLLM predicts.
            mllm_integration_strategy (str): The integration strategy of mLLM and LLM, defaults to "integration", options are "integration", "llm_only" and "mllm_only".
            chat_bot_config (dict): The parameters for LLM chatbot, including api_type, api_key... refer to config file for more details.
            retriever_config (dict): The parameters for LLM retriever, including api_type, api_key... refer to config file for more details.
        Returns:
            dict: A dictionary containing the chat results.
        """

        key_list = self.format_key(key_list)
        key_list_ori = key_list.copy()
        if len(key_list) == 0:
            return {"chat_res": "Error:输入的key_list无效！"}

        if not isinstance(visual_info, list):
            visual_info_list = [visual_info]
        else:
            visual_info_list = visual_info

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

        all_visual_info = self.merge_visual_info_list(visual_info_list)
        (
            all_normal_text_list,
            all_table_text_list,
            all_table_html_list,
            all_table_nei_text_list,
        ) = all_visual_info

        final_results = {}
        failed_results = ["大模型调用失败", "未知", "未找到关键信息", "None", ""]

        if len(key_list) > 0:
            related_text = self.get_related_normal_text(
                retriever_config,
                use_vector_retrieval,
                vector_info,
                key_list,
                all_normal_text_list,
                min_characters,
            )

            if len(related_text) > 0:
                prompt = self.text_pe.generate_prompt(
                    related_text,
                    key_list,
                    task_description=text_task_description,
                    output_format=text_output_format,
                    rules_str=text_rules_str,
                    few_shot_demo_text_content=text_few_shot_demo_text_content,
                    few_shot_demo_key_value_list=text_few_shot_demo_key_value_list,
                )
                self.generate_and_merge_chat_results(
                    chat_bot, prompt, key_list, final_results, failed_results
                )

        if len(key_list) > 0:
            for table_html, table_text, table_nei_text in zip(
                all_table_html_list, all_table_text_list, all_table_nei_text_list
            ):
                if len(table_html) <= min_characters - self.table_structure_len_max:
                    for table_info in [table_html]:
                        if len(key_list) > 0:

                            if len(table_nei_text) > 0:
                                table_info = (
                                    table_info + "\n 表格周围文字：" + table_nei_text
                                )

                            prompt = self.table_pe.generate_prompt(
                                table_info,
                                key_list,
                                task_description=table_task_description,
                                output_format=table_output_format,
                                rules_str=table_rules_str,
                                few_shot_demo_text_content=table_few_shot_demo_text_content,
                                few_shot_demo_key_value_list=table_few_shot_demo_key_value_list,
                            )

                            self.generate_and_merge_chat_results(
                                chat_bot,
                                prompt,
                                key_list,
                                final_results,
                                failed_results,
                            )

        if (
            self.use_mllm_predict
            and mllm_integration_strategy != "llm_only"
            and mllm_predict_info is not None
        ):
            if mllm_integration_strategy == "integration":
                final_predict_dict = self.ensemble_ocr_llm_mllm(
                    key_list_ori, final_results, mllm_predict_info
                )
            elif mllm_integration_strategy == "mllm_only":
                final_predict_dict = mllm_predict_info
            else:
                return {
                    "chat_res": f"Error:Unsupported mllm_integration_strategy {mllm_integration_strategy}, only support 'integration', 'llm_only' and 'mllm_only'!"
                }
        else:
            final_predict_dict = final_results
        return {"chat_res": final_predict_dict}

    def predict(self, *args, **kwargs) -> None:
        logging.error(
            "PP-ChatOCRv4-doc Pipeline do not support to call `predict()` directly! Please invoke `visual_predict`, `build_vector`, `chat` sequentially to obtain the result."
        )
        return
