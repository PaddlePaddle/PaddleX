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

import base64
import json
import re
from typing import Dict

from .....utils import logging
from .....utils.deps import class_requires_deps
from .base import BaseChat


@class_requires_deps("openai")
class OpenAIBotChat(BaseChat):
    """OpenAI Bot Chat"""

    entities = [
        "openai",
    ]

    def __init__(self, config: Dict) -> None:
        """Initializes the OpenAIBotChat with given configuration.

        Args:
            config (Dict): Configuration dictionary containing model_name, api_type, base_url, api_key, end_point.

        Raises:
            ValueError: If api_type is not one of ['openai'],
            base_url is None for api_type is openai,
            api_key is None for api_type is openai.
            ValueError: If end_point is not one of ['completion', 'chat_completion'].
        """
        from openai import OpenAI

        super().__init__()
        model_name = config.get("model_name", None)
        # compatible with historical model name
        if model_name == "ernie-3.5":
            model_name = "ernie-3.5-8k"
        api_type = config.get("api_type", None)
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)
        end_point = config.get("end_point", "chat_completion")

        if api_type not in ["openai"]:
            raise ValueError("api_type must be one of ['openai']")

        if api_type == "openai" and api_key is None:
            raise ValueError("api_key cannot be empty when api_type is openai.")

        if base_url is None:
            raise ValueError("base_url cannot be empty when api_type is openai.")

        if end_point not in ["completion", "chat_completion"]:
            raise ValueError(
                "end_point must be one of ['completion', 'chat_completion']"
            )

        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.model_name = model_name
        self.config = config

    def generate_chat_results(
        self,
        prompt: str,
        image: base64 = None,
        temperature: float = 0.001,
        max_retries: int = 1,
    ) -> Dict:
        """
        Generate chat results using the specified model and configuration.

        Args:
            prompt (str): The user's input prompt.
            image (base64): The user's input image for MLLM, defaults to None.
            temperature (float, optional): The temperature parameter for llms, defaults to 0.001.
            max_retries (int, optional): The maximum number of retries for llms API calls, defaults to 1.

        Returns:
            Dict: The chat completion result from the model.
        """
        llm_result = {"content": None, "reasoning_content": None}
        try:
            if image:
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            # XXX: give a basic prompt for common
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image}"
                                    },
                                },
                            ],
                        },
                    ],
                    stream=False,
                    temperature=temperature,
                    top_p=0.001,
                )
                llm_result["content"] = chat_completion.choices[0].message.content
                return llm_result
            elif self.config.get("end_point", "chat_completion") == "chat_completion":
                chat_completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    stream=False,
                    temperature=temperature,
                    top_p=0.001,
                )
                llm_result["content"] = chat_completion.choices[0].message.content
                try:
                    llm_result["reasoning_content"] = chat_completion.choices[
                        0
                    ].message.reasoning_content
                except:
                    pass
                return llm_result
            else:
                chat_completion = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=self.config.get("max_tokens", 1024),
                    temperature=float(temperature),
                    stream=False,
                )
                if isinstance(chat_completion, str):
                    chat_completion = json.loads(chat_completion)
                    llm_result = chat_completion["choices"][0]["text"]
                else:
                    llm_result["content"] = chat_completion.choices[0].text
                return llm_result
        except Exception as e:
            logging.error(e)
            self.ERROR_MASSAGE = "大模型调用失败"
        return llm_result

    def fix_llm_result_format(self, llm_result: str) -> dict:
        """
        Fix the format of the LLM result.

        Args:
            llm_result (str): The result from the LLM (Large Language Model).

        Returns:
            dict: A fixed format dictionary from the LLM result.
        """
        if not llm_result:
            return {}

        if "json" in llm_result or "```" in llm_result:
            index = llm_result.find("{")
            if index != -1:
                llm_result = llm_result[index:]
            index = llm_result.rfind("}")
            if index != -1:
                llm_result = llm_result[: index + 1]
            llm_result = (
                llm_result.replace("```", "").replace("json", "").replace("/n", "")
            )
            llm_result = llm_result.replace("[", "").replace("]", "")

        try:
            llm_result = json.loads(llm_result)
            llm_result_final = {}
            if "问题" in llm_result.keys() and "答案" in llm_result.keys():
                key = llm_result["问题"]
                value = llm_result["答案"]
                if isinstance(value, list):
                    if len(value) > 0:
                        llm_result_final[key] = value[0].strip(f"{key}:").strip(key)
                else:
                    llm_result_final[key] = value.strip(f"{key}:").strip(key)
                return llm_result_final
            for key in llm_result:
                value = llm_result[key]
                if isinstance(value, list):
                    if len(value) > 0:
                        llm_result_final[key] = value[0]
                else:
                    llm_result_final[key] = value
            return llm_result_final

        except:
            results = (
                llm_result.replace("\n", "")
                .replace("    ", "")
                .replace("{", "")
                .replace("}", "")
            )
            if not results.endswith('"'):
                results = results + '"'
            pattern = r'"(.*?)": "([^"]*)"'
            matches = re.findall(pattern, str(results))
            if len(matches) > 0:
                llm_result = {k: v for k, v in matches}
                if "问题" in llm_result.keys() and "答案" in llm_result.keys():
                    llm_result_final = {}
                    key = llm_result["问题"]
                    value = llm_result["答案"]
                    if isinstance(value, list):
                        if len(value) > 0:
                            llm_result_final[key] = value[0].strip(f"{key}:").strip(key)
                    else:
                        llm_result_final[key] = value.strip(f"{key}:").strip(key)
                    return llm_result_final
                return llm_result
            else:
                return {}
