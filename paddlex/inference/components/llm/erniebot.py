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

import os
import time
import json
import erniebot

from pathlib import Path
from .base import BaseLLM
from ....utils import logging
from ....utils.func_register import FuncRegister

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community import vectorstores
from erniebot_agent.extensions.langchain.embeddings import ErnieEmbeddings

__all__ = ["ErnieBot"]


class ErnieBot(BaseLLM):

    INPUT_KEYS = ["prompts"]
    OUTPUT_KEYS = ["cls_res"]
    DEAULT_INPUTS = {"prompts": "prompts"}
    DEAULT_OUTPUTS = {"cls_pred": "cls_pred"}
    API_TYPE = "aistudio"

    entities = [
        "ernie-4.0",
        "ernie-3.5",
        "ernie-3.5-8k",
        "ernie-lite",
        "ernie-tiny-8k",
        "ernie-speed",
        "ernie-speed-128k",
        "ernie-char-8k",
    ]

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(self, model_name="ernie-4.0", params={}):
        super().__init__()
        access_token = params.get("access_token")
        ak = params.get("ak")
        sk = params.get("sk")
        api_type = params.get("api_type")
        max_retries = params.get("max_retries")
        assert model_name in self.entities, f"model_name must be in {self.entities}"
        assert any([access_token, ak, sk]), "access_token or ak and sk must be set"
        self.model_name = model_name
        self.config = {
            "api_type": api_type,
            "max_retries": max_retries,
        }
        if access_token:
            self.config["access_token"] = access_token
        else:
            self.config["ak"] = ak
            self.config["sk"] = sk

    def pred(self, prompt, temperature=0.001):
        """
        llm predict
        """
        try:
            chat_completion = erniebot.ChatCompletion.create(
                _config_=self.config,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(temperature),
            )
            llm_result = chat_completion.get_result()
            return llm_result
        except Exception as e:
            if len(e.args) < 1:
                self.ERROR_MASSAGE = (
                    "当前选择后端为AI Studio，千帆调用失败，请检查token"
                )
            elif (
                e.args[-1]
                == "暂无权限使用，请在 AI Studio 正确获取访问令牌(access token)使用"
            ):
                self.ERROR_MASSAGE = (
                    "当前选择后端为AI Studio，请正确获取访问令牌(access token)使用"
                )
            elif e.args[-1] == "the max length of current question is 4800":
                self.ERROR_MASSAGE = "大模型调用失败"
            else:
                logging.error(e)
                self.ERROR_MASSAGE = "大模型调用失败"
        return None

    def get_vector(
        self,
        ocr_result,
        sleep_time=0.5,
        block_size=300,
        separators=["\t", "\n", "。", "\n\n", ""],
    ):
        """get summary prompt"""

        all_items = []
        for i, ocr_res in enumerate(ocr_result):
            for type, text in ocr_res.items():
                all_items += [f"第{i}页{type}：{text}"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=block_size, chunk_overlap=20, separators=separators
        )
        texts = text_splitter.split_text("\t".join(all_items))

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

        vectorstore = self.encode_vector_store(vectorstore.serialize_to_bytes())
        return vectorstore

    def caculate_similar(self, vector, key_list, llm_params=None, sleep_time=0.5):
        """caculate similar with key and doc"""
        if self.is_vector_store(vector):
            # XXX: The initialization parameters are hard-coded.
            if llm_params:
                api_type = llm_params.get("api_type")
                access_token = llm_params.get("access_token")
                ak = llm_params.get("ak")
                sk = llm_params.get("sk")
            else:
                api_type = self.config["api_type"]
                access_token = self.config.get("access_token")
                ak = self.config.get("ak")
                sk = self.config.get("sk")
            if api_type == "aistudio":
                embeddings = ErnieEmbeddings(aistudio_access_token=access_token)
            elif api_type == "qianfan":
                embeddings = QianfanEmbeddingsEndpoint(qianfan_ak=ak, qianfan_sk=sk)
            else:
                raise ValueError(f"Unsupported api_type: {api_type}")

        vectorstore = vectorstores.FAISS.deserialize_from_bytes(
            self.decode_vector_store(vector), embeddings
        )

        # 根据提问匹配上下文
        Q = []
        C = []
        for key in key_list:
            QUESTION = f"抽取关键信息:{key}"
            # c_str = ""
            Q.append(QUESTION)
            time.sleep(sleep_time)
            docs = vectorstore.similarity_search_with_relevance_scores(QUESTION, k=2)
            context = [(document.page_content, score) for document, score in docs]
            context = sorted(context, key=lambda x: x[1])
            C.extend([x[0] for x in context[::-1]])

        C = list(set(C))
        all_C = " ".join(C)

        summary_prompt = all_C

        return summary_prompt
