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
from abc import ABC, abstractmethod

import time
import base64

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community import vectorstores

from paddlex.utils import logging

from .....utils.subclass_register import AutoRegisterABCMetaClass


class BaseRetriever(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Retriever"""

    __is_base = True

    VECTOR_STORE_PREFIX = "PADDLEX_VECTOR_STORE"

    def __init__(self):
        """Initializes an instance of base retriever."""
        super().__init__()
        self.model_name = None
        self.embedding = None

    @abstractmethod
    def generate_vector_database(self):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of generate_vector_database.
        """
        raise NotImplementedError(
            "The method `generate_vector_database` has not been implemented yet."
        )

    @abstractmethod
    def similarity_retrieval(self):
        """
        Declaration of an abstract method. Subclasses are expected to
        provide a concrete implementation of similarity_retrieval.
        """
        raise NotImplementedError(
            "The method `similarity_retrieval` has not been implemented yet."
        )

    def get_model_name(self) -> str:
        """
        Get the model name used for generating vectors.

        Returns:
            str: The model name.
        """
        return self.model_name

    def is_vector_store(self, s: str) -> bool:
        """
        Check if the given string starts with the vector store prefix.

        Args:
            s (str): The input string to check.

        Returns:
            bool: True if the string starts with the vector store prefix, False otherwise.
        """
        return s.startswith(self.VECTOR_STORE_PREFIX)

    def encode_vector_store(self, vector_store_bytes: bytes) -> str:
        """
        Encode the vector store bytes into a base64 string prefixed with a specific prefix.

        Args:
            vector_store_bytes (bytes): The bytes to encode.

        Returns:
            str: The encoded string with the prefix.
        """
        return self.VECTOR_STORE_PREFIX + base64.b64encode(vector_store_bytes).decode(
            "ascii"
        )

    def decode_vector_store(self, vector_store_str: str) -> bytes:
        """
        Decodes the vector store string by removing the prefix and decoding the base64 encoded string.

        Args:
            vector_store_str (str): The vector store string with a prefix.

        Returns:
            bytes: The decoded vector store data.
        """
        return base64.b64decode(vector_store_str[len(self.VECTOR_STORE_PREFIX) :])

    def generate_vector_database(
        self,
        text_list: List[str],
        block_size: int = 300,
        separators: List[str] = ["\t", "\n", "。", "\n\n", ""],
    ) -> FAISS:
        """
        Generates a vector database from a list of texts.

        Args:
            text_list (list[str]): A list of texts to generate the vector database from.
            block_size (int): The size of each chunk to split the text into.
            separators (list[str]): A list of separators to use when splitting the text.

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

        try:
            vectorstore = FAISS.from_documents(
                documents=all_splits, embedding=self.embedding
            )
        except ValueError as e:
            print(e)
            vectorstore = None

        return vectorstore

    def encode_vector_store_to_bytes(self, vectorstore: FAISS) -> str:
        """
        Encode the vector store serialized to bytes.

        Args:
            vectorstore (FAISS): The vector store to be serialized and encoded.

        Returns:
            str: The encoded vector store.
        """
        if vectorstore is None:
            vectorstore = self.VECTOR_STORE_PREFIX
        else:
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

        vectorstore = self.decode_vector_store(vectorstore)

        if vectorstore == b"":
            logging.warning("The retrieved vectorstore is empty,will empty vector.")
            return None

        print(vectorstore)

        vector = vectorstores.FAISS.deserialize_from_bytes(
            vectorstore,
            embeddings=self.embedding,
            allow_dangerous_deserialization=True,
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
        if vectorstore is None:
            return all_C
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
