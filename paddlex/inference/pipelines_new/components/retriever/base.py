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

from abc import ABC, abstractmethod
import inspect
import base64
from .....utils.subclass_register import AutoRegisterABCMetaClass


class BaseRetriever(ABC, metaclass=AutoRegisterABCMetaClass):
    """Base Retriever"""

    __is_base = True

    VECTOR_STORE_PREFIX = "PADDLEX_VECTOR_STORE"

    def __init__(self):
        """Initializes an instance of base retriever."""
        super().__init__()

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
