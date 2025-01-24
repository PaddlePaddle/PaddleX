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

import re
import json
import erniebot
from typing import Dict
from .....utils import logging
from .base import BaseChat


class ErnieBotChat(BaseChat):
    """Ernie Bot Chat"""

    entities = [
        "aistudio",
        "qianfan",
    ]

    MODELS = [
        "ernie-4.0",
        "ernie-3.5",
        "ernie-3.5-8k",
        "ernie-lite",
        "ernie-tiny-8k",
        "ernie-speed",
        "ernie-speed-128k",
        "ernie-char-8k",
    ]

    def __init__(self, config: Dict) -> None:
        """Initializes the ErnieBotChat with given configuration.

        Args:
            config (Dict): Configuration dictionary containing model_name, api_type, ak, sk, and access_token.

        Raises:
            ValueError: If model_name is not in the predefined entities,
            api_type is not one of ['aistudio', 'qianfan'],
            access_token is None for 'aistudio' api_type,
            or ak and sk are None for 'qianfan' api_type.
        """
        super().__init__()
        model_name = config.get("model_name", None)
        api_type = config.get("api_type", None)
        ak = config.get("ak", None)
        sk = config.get("sk", None)
        access_token = config.get("access_token", None)

        if model_name not in self.MODELS:
            raise ValueError(f"model_name must be in {self.MODELS} of ErnieBotChat.")

        if api_type not in ["aistudio", "qianfan"]:
            raise ValueError("api_type must be one of ['aistudio', 'qianfan']")

        if api_type == "aistudio" and access_token is None:
            raise ValueError("access_token cannot be empty when api_type is aistudio.")

        if api_type == "qianfan" and (ak is None or sk is None):
            raise ValueError("ak and sk cannot be empty when api_type is qianfan.")

        self.model_name = model_name
        self.config = config

    def generate_chat_results(
        self, prompt: str, temperature: float = 0.001, max_retries: int = 1
    ) -> Dict:
        """
        Generate chat results using the specified model and configuration.

        Args:
            prompt (str): The user's input prompt.
            temperature (float, optional): The temperature parameter for llms, defaults to 0.001.
            max_retries (int, optional): The maximum number of retries for llms API calls, defaults to 1.

        Returns:
            Dict: The chat completion result from the model.
        """
        try:
            cur_config = {
                "api_type": self.config["api_type"],
                "max_retries": max_retries,
            }
            if self.config["api_type"] == "aistudio":
                cur_config["access_token"] = self.config["access_token"]
            elif self.config["api_type"] == "qianfan":
                cur_config["ak"] = self.config["ak"]
                cur_config["sk"] = self.config["sk"]
            chat_completion = erniebot.ChatCompletion.create(
                _config_=cur_config,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
            )
            llm_result = chat_completion.get_result()
            return llm_result
        except Exception as e:
            if len(e.args) < 1:
                self.ERROR_MASSAGE = "暂无权限访问ErnieBot服务，请检查访问令牌。"
            elif (
                e.args[-1]
                == "暂无权限使用，请在 AI Studio 正确获取访问令牌(access token)使用"
            ):
                self.ERROR_MASSAGE = "暂无权限访问ErnieBot服务，请检查访问令牌。"
            else:
                logging.error(e)
                self.ERROR_MASSAGE = "大模型调用失败"
        return None

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
