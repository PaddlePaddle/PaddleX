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

import json
from typing import Dict, List

import requests

from paddlex.utils import logging

from .....utils.deps import is_dep_available
from .base import BaseRetriever


class QianFanBotRetriever(BaseRetriever):
    """QianFan Bot Retriever"""

    entities = [
        "qianfan",
    ]

    MODELS = [
        "tao-8k",
        "embedding-v1",
        "bge-large-zh",
        "bge-large-en",
    ]

    def __init__(self, config: Dict) -> None:
        """
        Initializes the ErnieBotRetriever instance with the provided configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.
                - model_name (str): The name of the model to use.
                - api_type (str): The type of API to use ('qianfan' or 'openai').
                - api_key (str): The API key for 'qianfan' API.
                - base_url (str): The base URL for 'qianfan' API.

        Raises:
            ValueError: If api_type is not one of ['qianfan','openai'],
                base_url is None for api_type is qianfan,
                api_key is None for api_type is qianfan.
        """
        super().__init__()

        model_name = config.get("model_name", None)
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)

        if model_name not in self.MODELS:
            raise ValueError(
                f"model_name must be in {self.MODELS} of QianFanBotRetriever."
            )

        if api_key is None:
            raise ValueError("api_key cannot be empty when api_type is qianfan.")

        if base_url is None:
            raise ValueError("base_url cannot be empty when api_type is qianfan.")

        self.embedding = QianfanEmbeddings(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )

        self.model_name = model_name
        self.config = config


if is_dep_available("langchain-core"):
    from langchain_core.embeddings import Embeddings

    class QianfanEmbeddings(Embeddings):
        """`Baidu Qianfan Embeddings` embedding models."""

        def __init__(
            self,
            api_key: str,
            base_url: str = "https://qianfan.baidubce.com/v2",
            model: str = "embedding-v1",
            **kwargs,
        ):
            """
            Initialize the Baidu Qianfan Embeddings class.

            Args:
                api_key (str): The Qianfan API key.
                base_url (str): The base URL for 'qianfan' API.
                model (str): Model name. Default is "embedding-v1",select in ["tao-8k","embedding-v1","bge-large-en","bge-large-zh"].
                kwargs (dict): Additional keyword arguments passed to the base Embeddings class.
            """
            super().__init__(**kwargs)
            chunk_size_map = {
                "tao-8k": 1,
                "embedding-v1": 16,
                "bge-large-en": 16,
                "bge-large-zh": 16,
            }
            self.api_key = api_key
            self.base_url = base_url
            self.model = model
            self.chunk_size = chunk_size_map.get(model, 1)

        def embed(self, texts: str, **kwargs) -> List[float]:
            url = f"{self.base_url}/embeddings"
            payload = json.dumps(
                {"model": kwargs.get("model", self.model), "input": [f"{texts}"]}
            )
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code != 200:
                logging.error(
                    f"Failed to call Qianfan API. Status code: {response.status_code}, Response content: {response}"
                )

            return response.json()

        def embed_query(self, text: str) -> List[float]:
            resp = self.embed_documents([text])
            return resp[0]

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            """
            Embeds a list of text documents using the AutoVOT algorithm.

            Args:
                texts (List[str]): A list of text documents to embed.

            Returns:
                List[List[float]]: A list of embeddings for each document in the input list.
                                Each embedding is represented as a list of float values.
            """
            lst = []
            for chunk in texts:
                resp = self.embed(texts=chunk)
                lst.extend([res["embedding"] for res in resp["data"]])
            return lst

        async def aembed_query(self, text: str) -> List[float]:
            embeddings = await self.aembed_documents([text])
            return embeddings[0]

        async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
            lst = []
            for chunk in texts:
                resp = await self.embed(texts=chunk)
                for res in resp["data"]:
                    lst.extend([res["embedding"]])
            return lst
