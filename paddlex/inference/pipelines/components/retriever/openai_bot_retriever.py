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

from .base import BaseRetriever


class OpenAIBotRetriever(BaseRetriever):
    """OpenAI Bot Retriever"""

    entities = [
        "openai",
    ]

    def __init__(self, config: Dict) -> None:
        """
        Initializes the OpenAIBotRetriever instance with the provided configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.
                - model_name (str): The name of the model to use.
                - api_type (str): The type of API to use ('qianfan' or 'openai').
                - api_key (str): The API key for 'openai' API.
                - base_url (str): The base URL for 'openai' API.

        Raises:
            ValueError: If api_type is not one of ['qianfan','openai'],
            base_url is None for api_type is openai,
            api_key is None for api_type is openai.
        """
        super().__init__()

        model_name = config.get("model_name", None)
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)
        tiktoken_enabled = config.get("tiktoken_enabled", False)

        if api_key is None:
            raise ValueError("api_key cannot be empty when api_type is openai.")

        if base_url is None:
            raise ValueError("base_url cannot be empty when api_type is openai.")

        try:
            from langchain_openai import OpenAIEmbeddings
        except:
            raise Exception(
                "langchain-openai is not installed, please install it first."
            )

        self.embedding = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            tiktoken_enabled=tiktoken_enabled,
        )

        self.model_name = model_name
        self.config = config
