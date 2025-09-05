# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from paddlex_hps_server import BaseTritonPythonModel, schemas


class TritonPythonModel(BaseTritonPythonModel):
    @property
    def pipeline_creation_kwargs(self):
        return {"initial_predictor": False}

    def get_input_model_type(self):
        return schemas.pp_chatocrv3_doc.ChatRequest

    def get_result_model_type(self):
        return schemas.pp_chatocrv3_doc.ChatResult

    def run(self, input, log_id):
        result = self.pipeline.chat(
            input.keyList,
            input.visualInfo,
            use_vector_retrieval=input.useVectorRetrieval,
            vector_info=input.vectorInfo,
            min_characters=input.minCharacters,
            text_task_description=input.textTaskDescription,
            text_output_format=input.textOutputFormat,
            text_rules_str=input.textRulesStr,
            text_few_shot_demo_text_content=input.textFewShotDemoTextContent,
            text_few_shot_demo_key_value_list=input.textFewShotDemoKeyValueList,
            table_task_description=input.tableTaskDescription,
            table_output_format=input.tableOutputFormat,
            table_rules_str=input.tableRulesStr,
            table_few_shot_demo_text_content=input.tableFewShotDemoTextContent,
            table_few_shot_demo_key_value_list=input.tableFewShotDemoKeyValueList,
            chat_bot_config=input.chatBotConfig,
            retriever_config=input.retrieverConfig,
        )

        return schemas.pp_chatocrv3_doc.ChatResult(
            chatResult=result["chat_res"],
        )
