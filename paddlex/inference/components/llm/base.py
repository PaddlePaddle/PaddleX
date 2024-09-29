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

import base64
from ..base import BaseComponent
from ....utils.subclass_register import AutoRegisterABCMetaClass

__all__ = ["BaseLLM"]


class BaseLLM(BaseComponent, metaclass=AutoRegisterABCMetaClass):
    __is_base = True

    ERROR_MASSAGE = ""
    VECTOR_STORE_PREFIX = "PADDLEX_VECTOR_STORE"

    def __init__(self):
        super().__init__()

    def pre_process(self, inputs):
        return inputs

    def post_process(self, outputs):
        return outputs

    def pred(self, inputs):
        raise NotImplementedError("The method `pred` has not been implemented yet.")

    def get_vector(self):
        raise NotImplementedError(
            "The method `get_vector` has not been implemented yet."
        )

    def caculate_similar(self):
        raise NotImplementedError(
            "The method `caculate_similar` has not been implemented yet."
        )

    def apply(self, inputs):
        pre_process_results = self.pre_process(inputs)
        pred_results = self.pred(pre_process_results)
        post_process_results = self.post_process(pred_results)
        return post_process_results

    def is_vector_store(self, s):
        return s.startswith(self.VECTOR_STORE_PREFIX)

    def encode_vector_store(self, vector_store_bytes):
        return self.VECTOR_STORE_PREFIX + base64.b64encode(vector_store_bytes).decode(
            "ascii"
        )

    def decode_vector_store(self, vector_store_str):
        return base64.b64decode(vector_store_str[len(self.VECTOR_STORE_PREFIX) :])
