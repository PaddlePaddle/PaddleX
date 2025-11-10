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

from typing import Any, Dict, List

from paddlex_hps_server import BaseTritonPythonModel, schemas


class TritonPythonModel(BaseTritonPythonModel):
    def get_input_model_type(self):
        return schemas.pp_doctranslation.TranslateRequest

    def get_result_model_type(self):
        return schemas.pp_doctranslation.TranslateResult

    def run(self, input, log_id):
        ori_md_info_list: List[Dict[str, Any]] = []
        for i, item in enumerate(input.markdownList):
            ori_md_info_list.append(
                {
                    "input_path": None,
                    "page_index": i,
                    "markdown_texts": item.text,
                    "page_continuation_flags": (item.isStart, item.isEnd),
                }
            )

        result = self.pipeline.translate(
            ori_md_info_list,
            target_language=input.targetLanguage,
            chunk_size=input.chunkSize,
            task_description=input.taskDescription,
            output_format=input.outputFormat,
            rules_str=input.rulesStr,
            few_shot_demo_text_content=input.fewShotDemoTextContent,
            few_shot_demo_key_value_list=input.fewShotDemoKeyValueList,
            glossary=input.glossary,
            llm_request_interval=input.llmRequestInterval,
            chat_bot_config=input.chatBotConfig,
        )

        translation_results: List[Dict[str, Any]] = []
        for item in result:
            translation_results.append(
                dict(
                    language=item["language"],
                    markdown=dict(
                        text=item["markdown_texts"],
                        isStart=item["page_continuation_flags"][0],
                        isEnd=item["page_continuation_flags"][1],
                    ),
                )
            )

        return schemas.pp_doctranslation.TranslateResult(
            translationResults=translation_results,
        )
