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

from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-DocTranslation")

input_path = "document_sample.pdf"
output_path = "./output"

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}


if input_path.lower().endswith(".md"):
    ori_md_info_list = pipeline.load_from_markdown(input_path)
else:
    # Use PP-StructureV3 to get original markdown info list
    visual_predict_res = pipeline.visual_predict(
        input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=True,
        use_table_recognition=True,
    )

    ori_md_info_list = []
    for res in visual_predict_res:
        layout_parsing_result = res["layout_parsing_result"]
        ori_md_info_list.append(layout_parsing_result.markdown)
        layout_parsing_result.save_to_img(output_path)
        layout_parsing_result.save_to_markdown(output_path)

    # To concatenate markdown pages into a single markdown file, when input is a pdf file
    if input_path.lower().endswith(".pdf"):
        ori_md_info = pipeline.concatenate_markdown_pages(ori_md_info_list)
        ori_md_info.save_to_markdown(output_path)

tgt_md_info_list = pipeline.translate(
    ori_md_info_list=ori_md_info_list,
    target_language="en",
    chunk_size=5000,
    chat_bot_config=chat_bot_config,
)
for tgt_md_info in tgt_md_info_list:
    tgt_md_info.save_to_markdown(output_path)
