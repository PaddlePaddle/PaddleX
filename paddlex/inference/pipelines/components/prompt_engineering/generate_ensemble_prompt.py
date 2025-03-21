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


class GenerateEnsemblePrompt(BaseGeneratePrompt):
    """Generate Ensemble Prompt"""

    entities = ["ensemble_prompt"]

    def __init__(self, config: Dict) -> None:
        """Initializes the GenerateEnsemblePrompt instance with the given configuration.

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
        key: str,
        result_methodA: str,
        result_methodB: str,
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
        if task_description is None:
            task_description = self.task_description

        if output_format is None:
            output_format = self.output_format

        if rules_str is None:
            rules_str = self.rules_str

        if few_shot_demo_text_content is None:
            few_shot_demo_text_content = self.few_shot_demo_text_content

        if few_shot_demo_key_value_list is None:
            few_shot_demo_key_value_list = self.few_shot_demo_key_value_list

        prompt = f"""{task_description}{rules_str}{output_format}{few_shot_demo_text_content}{few_shot_demo_key_value_list}"""
        task_type = self.task_type
        if task_type == "ensemble_prompt":
            prompt += f"""下面正式开始:
                \n问题：```{key}```
                \n方法A的结果：{result_methodA}
                \n方法B的结果：{result_methodB}
                """
        else:
            raise ValueError(f"{self.task_type} is currently not supported.")
        return prompt
