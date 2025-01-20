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

from .base import BaseRetriever

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community import vectorstores

import time

from typing import Dict


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
                - api_type (str): The type of API to use ('aistudio', 'qianfan' or 'openai').
                - api_key (str, optional): The API key for 'openai' API.
                - base_url (str, optional): The base URL for 'openai' API.

        Raises:
            ValueError: If api_type is not one of ['openai'],
            base_url is None for api_type is openai,
            api_key is None for api_type is openai.
        """
        super().__init__()

        model_name = config.get("model_name", None)
        api_type = config.get("api_type", None)
        api_key = config.get("api_key", None)
        base_url = config.get("base_url", None)
        tiktoken_enabled = config.get("tiktoken_enabled", False)

        if api_type not in ["openai"]:
            raise ValueError("api_type must be one of ['openai']")

        if api_type == "openai" and api_key is None:
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

    # Generates a vector database from a list of texts using different embeddings based on the configured API type.

    def generate_vector_database(
        self,
        text_list: list[str],
        block_size: int = 300,
        separators: list[str] = ["\t", "\n", "。", "\n\n", ""],
        sleep_time: float = 0.5,
    ) -> FAISS:
        """
        Generates a vector database from a list of texts.

        Args:
            text_list (list[str]): A list of texts to generate the vector database from.
            block_size (int): The size of each chunk to split the text into.
            separators (list[str]): A list of separators to use when splitting the text.
            sleep_time (float): The time to sleep between embedding generations to avoid rate limiting.

        Returns:
            FAISS: The generated vector database.

        Raises:
            ValueError: If an unsupported API type is configured.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=block_size, chunk_overlap=20, separators=separators
        )
        texts = text_splitter.split_text("\t".join(text_list))
        all_splits = [Document(page_content=text) for text in texts]

        api_type = self.config["api_type"]

        vectorstore = FAISS.from_documents(
            documents=all_splits, embedding=self.embedding
        )

        return vectorstore

    def encode_vector_store_to_bytes(self, vectorstore: FAISS) -> str:
        """
        Encode the vector store serialized to bytes.

        Args:
            vectorstore (FAISS): The vector store to be serialized and encoded.

        Returns:
            str: The encoded vector store.
        """
        vectorstore = self.encode_vector_store(vectorstore.serialize_to_bytes())
        return vectorstore

    def decode_vector_store_from_bytes(self, vectorstore: str) -> FAISS:
        """
        Decode a vector store from bytes according to the specified API type.

        Args:
            vectorstore (str): The serialized vector store string.

        Returns:
            FAISS: Deserialized vector store object.

        Raises:
            ValueError: If the retrieved vector store is not for PaddleX
            or if an unsupported API type is specified.
        """
        if not self.is_vector_store(vectorstore):
            raise ValueError("The retrieved vectorstore is not for PaddleX.")

        vector = vectorstores.FAISS.deserialize_from_bytes(
            self.decode_vector_store(vectorstore), self.embedding
        )
        return vector

    def similarity_retrieval(
        self, query_text_list: list[str], vectorstore: FAISS, sleep_time: float = 0.5
    ) -> str:
        """
        Retrieve similar contexts based on a list of query texts.

        Args:
            query_text_list (list[str]): A list of query texts to search for similar contexts.
            vectorstore (FAISS): The vector store where to perform the similarity search.
            sleep_time (float): The time to sleep between each query, in seconds. Default is 0.5.

        Returns:
            str: A concatenated string of all unique contexts found.
        """
        C = []
        for query_text in query_text_list:
            QUESTION = query_text
            time.sleep(sleep_time)
            docs = vectorstore.similarity_search_with_relevance_scores(QUESTION, k=2)
            context = [(document.page_content, score) for document, score in docs]
            context = sorted(context, key=lambda x: x[1])
            C.extend([x[0] for x in context[::-1]])
        C = list(set(C))
        all_C = " ".join(C)
        return all_C
