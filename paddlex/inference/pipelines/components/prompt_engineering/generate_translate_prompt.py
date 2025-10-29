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

from typing import Dict

from .base import BaseGeneratePrompt


class GenerateTranslatePrompt(BaseGeneratePrompt):
    """Generate Ensemble Prompt"""

    entities = ["translate_prompt"]

    def __init__(self, config: Dict) -> None:
        """Initializes the GenerateTranslatePrompt instance with the given configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.
                - task_type (str): The type of task to generate a prompt for, in the support entities list.
                - task_description (str, optional): A description of the task. Defaults to an empty string.
                - output_format (str, optional): The desired output format. Defaults to an empty string.
                - rules_str (str, optional): A string representing rules for the task. Defaults to an empty string.
                - few_shot_demo_text_content (str, optional): Text content for few-shot demos. Defaults to an empty string.
                - few_shot_demo_key_value_list (str, optional): A key-value list for few-shot demos. Defaults to an empty string.

        Raises:
            ValueError: If the task type is not in the allowed entities for GenerateKIEPrompt.
        """
        super().__init__()

        task_type = config.get("task_type", "")
        task_description = config.get("task_description", "")
        output_format = config.get("output_format", "")
        rules_str = config.get("rules_str", "")
        few_shot_demo_text_content = config.get("few_shot_demo_text_content", "")
        few_shot_demo_key_value_list = config.get("few_shot_demo_key_value_list", "")

        if task_description is None:
            task_description = ""

        if output_format is None:
            output_format = ""

        if rules_str is None:
            rules_str = ""

        if few_shot_demo_text_content is None:
            few_shot_demo_text_content = ""

        if few_shot_demo_key_value_list is None:
            few_shot_demo_key_value_list = ""

        if task_type not in self.entities:
            raise ValueError(
                f"task type must be in {self.entities} of GenerateEnsemblePrompt."
            )

        self.task_type = task_type
        self.task_description = task_description
        self.output_format = output_format
        self.rules_str = rules_str
        self.few_shot_demo_text_content = few_shot_demo_text_content
        self.few_shot_demo_key_value_list = few_shot_demo_key_value_list

    def generate_prompt(
        self,
        original_text: str,
        language: str,
        task_description: str = None,
        output_format: str = None,
        rules_str: str = None,
        few_shot_demo_text_content: str = None,
        few_shot_demo_key_value_list: str = None,
    ) -> str:
        """Generates a prompt based on the given parameters.
        Args:
            key (str): the input question.
            result_methodA (str): the result of method A.
            result_methodB (str): the result of method B.
            task_description (str, optional): A description of the task. Defaults to None.
            output_format (str, optional): The desired output format. Defaults to None.
            rules_str (str, optional): A string containing rules or instructions. Defaults to None.
            few_shot_demo_text_content (str, optional): Text content for few-shot demos. Defaults to None.
            few_shot_demo_key_value_list (str, optional): Key-value list for few-shot demos. Defaults to None.
        Returns:
            str: The generated prompt.

        Raises:
            ValueError: If the task_type is not supported.
        """
        language_map = {
            "chinese": "简体中文",
            "zh": "简体中文",
            "english": "英语",
            "en": "英语",
            "french": "法语",
            "fr": "法语",
            "spanish": "西班牙语",
            "es": "西班牙语",
            "german": "德语",
            "de": "德语",
            "japanese": "日语",
            "ja": "日语",
            "korean": "韩语",
            "ko": "韩语",
            "russian": "俄语",
            "ru": "俄语",
            "italian": "意大利语",
            "it": "意大利语",
            "portuguese": "葡萄牙语",
            "pt": "葡萄牙语",
            "arabic": "阿拉伯语",
            "ar": "阿拉伯语",
            "hindi": "印地语",
            "hi": "印地语",
            "dutch": "荷兰语",
            "nl": "荷兰语",
            "swedish": "瑞典语",
            "sv": "瑞典语",
            "turkish": "土耳其语",
            "tr": "土耳其语",
            "thai": "泰语",
            "th": "泰语",
            "vietnamese": "越南语",
            "vi": "越南语",
            "hebrew": "希伯来语",
            "he": "希伯来语",
            "greek": "希腊语",
            "el": "希腊语",
            "polish": "波兰语",
            "pl": "波兰语",
        }

        if task_description is None:
            task_description = self.task_description

        if output_format is None:
            output_format = self.output_format

        if rules_str is None:
            rules_str = self.rules_str

        if few_shot_demo_text_content is None:
            few_shot_demo_text_content = self.few_shot_demo_text_content

        if few_shot_demo_text_content:
            few_shot_demo_text_content = (
                f"这里是一些示例：\n{few_shot_demo_text_content}\n"
            )

        if few_shot_demo_key_value_list is None:
            few_shot_demo_key_value_list = self.few_shot_demo_key_value_list

        if few_shot_demo_key_value_list:
            few_shot_demo_key_value_list = f"\n这里是一些专业术语对照表,如果遇到对照表中单词要参考对照表翻译：\n{few_shot_demo_key_value_list}\n"

        after_rule = "9. 请在翻译完成后添加特殊标记 <<END>>，确保翻译完整。"
        prompt = f"""{task_description}{rules_str}{after_rule}{output_format}{few_shot_demo_text_content}{few_shot_demo_key_value_list}"""
        
        language_name = language_map.get(language, language)
        task_type = self.task_type
        if task_type == "translate_prompt":
            prompt += f"""下面正式开始:
                \n将以下内容翻译成：{language_name}
                \n原文：{original_text}
                """
        else:
            raise ValueError(f"{self.task_type} is currently not supported.")
        return prompt
