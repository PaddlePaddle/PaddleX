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
# # -*- coding: utf-8 -*-

from paddlex import create_pipeline
import os
from pdf_to_md_to_word import json_to_html_with_headfoot
from pdf_to_md_to_latex import md_to_latex

input_path = "mypaddle/input/lunwen_cn_n.pdf"
output_path = "mypaddle/upgit/output"

pipeline = create_pipeline(pipeline="PP-DocTranslation")

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-4.5-turbo-128k-preview",
    "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    "api_type": "openai",
    "api_key": "e89bbf05ca61a58ad60a7493e8657649088508dc",
}

if input_path.lower().endswith(".md"):
    ori_md_info_list = pipeline.load_from_markdown(input_path)
else:
    visual_predict_res = pipeline.visual_predict(
        input_path,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_common_ocr=True,
        use_seal_recognition=False,
        use_table_recognition=True,
    )

    ori_md_info_list = []
    json_list = []

    for res in visual_predict_res:
        layout_parsing_result = res["layout_parsing_result"]
        
        # 获取每页的解析json
        json_data = layout_parsing_result.json
        json_list.append(json_data)
        
        layout_parsing_result.save_to_markdown(output_path)
        md_data = layout_parsing_result.markdown
        ori_md_info_list.append(md_data)
    
    # 获取解析之后json中的页眉页脚，放入md的后面，一同送入模型进行翻译  
    head_foot_table = json_to_html_with_headfoot(json_list,input_path)
    ori_md_info_list.append(head_foot_table)
    
    if input_path.lower().endswith(".pdf"):
        ori_md_info = pipeline.concatenate_markdown_pages(ori_md_info_list)
        ori_md_info.save_to_markdown(output_path)
 
tgt_md_info_list = pipeline.translate(
    ori_md_info_list=ori_md_info_list,
    target_language="en",
    chunk_size=3000,
    chat_bot_config=chat_bot_config,
    # 修改translate中调用的concatenate_markdown_pages函数。默认为false
    use_flags = True,
)


for tgt_md_info in tgt_md_info_list:
    tgt_md_info.save_to_markdown(output_path)

# 调用转word函数
base_name = os.path.splitext(os.path.basename(input_path))[0]
     
with open(f"{output_path}/{base_name}_en.md", "r", encoding="utf-8") as f:
    md_text = f.read()

md_to_latex(md_text, f"{output_path}/{base_name}_md2latex.tex")
