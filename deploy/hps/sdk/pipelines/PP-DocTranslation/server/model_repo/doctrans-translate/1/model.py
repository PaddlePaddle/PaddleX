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
