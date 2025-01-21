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

from typing import Dict, List
import time
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community import vectorstores
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings
from .base import BaseRetriever


class ErnieBotRetriever(BaseRetriever):
    """Ernie Bot Retriever"""

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
        """
        Initializes the ErnieBotRetriever instance with the provided configuration.

        Args:
            config (Dict): A dictionary containing configuration settings.
                - model_name (str): The name of the model to use.
                - api_type (str): The type of API to use ('aistudio', 'qianfan' or 'openai').
                - ak (str, optional): The access key for 'qianfan' API.
                - sk (str, optional): The secret key for 'qianfan' API.
                - access_token (str, optional): The access token for 'aistudio' API.

        Raises:
            ValueError: If model_name is not in self.entities,
                api_type is not 'aistudio' or 'qianfan',
                access_token is missing for 'aistudio' API,
                or ak and sk are missing for 'qianfan' API.
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

    # Generates a vector database from a list of texts using different embeddings based on the configured API type.

    def generate_vector_database(
        self,
        text_list: List[str],
        block_size: int = 300,
        separators: List[str] = ["\t", "\n", "。", "\n\n", ""],
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
        if api_type == "qianfan":
            os.environ["QIANFAN_AK"] = os.environ.get("EB_AK", self.config["ak"])
            os.environ["QIANFAN_SK"] = os.environ.get("EB_SK", self.config["sk"])
            user_ak = os.environ.get("EB_AK", self.config["ak"])
            user_id = hash(user_ak)
            vectorstore = FAISS.from_documents(
                documents=all_splits, embedding=QianfanEmbeddingsEndpoint()
            )
        elif api_type == "aistudio":
            token = self.config["access_token"]
            vectorstore = FAISS.from_documents(
                documents=all_splits[0:1],
                embedding=ErnieEmbeddings(aistudio_access_token=token),
            )
            #### ErnieEmbeddings.chunk_size = 16
            step = min(16, len(all_splits) - 1)
            for shot_splits in [
                all_splits[i : i + step] for i in range(1, len(all_splits), step)
            ]:
                time.sleep(sleep_time)
                vectorstore_slice = FAISS.from_documents(
                    documents=shot_splits,
                    embedding=ErnieEmbeddings(aistudio_access_token=token),
                )
                vectorstore.merge_from(vectorstore_slice)
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

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

        api_type = self.config["api_type"]

        if api_type == "aistudio":
            access_token = self.config["access_token"]
            embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
        elif api_type == "qianfan":
            ak = self.config["ak"]
            sk = self.config["sk"]
            embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=ak, qianfan_sk=sk)
        else:
            raise ValueError(f"Unsupported api_type: {api_type}")

        vector = vectorstores.FAISS.deserialize_from_bytes(
            self.decode_vector_store(vectorstore), embeddings
        )
        return vector

    def similarity_retrieval(
        self,
        query_text_list: List[str],
        vectorstore: FAISS,
        sleep_time: float = 0.5,
        topk: int = 2,
        min_characters: int = 3500,
    ) -> str:
        """
        Retrieve similar contexts based on a list of query texts.

        Args:
            query_text_list (list[str]): A list of query texts to search for similar contexts.
            vectorstore (FAISS): The vector store where to perform the similarity search.
            sleep_time (float): The time to sleep between each query, in seconds. Default is 0.5.
            topk (int): The number of results to retrieve per query. Default is 2.
            min_characters (int): The minimum number of characters required for text processing, defaults to 3500.
        Returns:
            str: A concatenated string of all unique contexts found.
        """
        C = []
        all_C = ""
        for query_text in query_text_list:
            QUESTION = query_text
            time.sleep(sleep_time)
            docs = vectorstore.similarity_search_with_relevance_scores(QUESTION, k=topk)
            context = [(document.page_content, score) for document, score in docs]
            context = sorted(context, key=lambda x: x[1])
            for text, score in context[::-1]:
                if score >= -0.1:
                    if len(all_C) + len(text) > min_characters:
                        break
                    all_C += text
        return all_C
