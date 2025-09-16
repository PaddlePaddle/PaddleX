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

# -*- coding: utf-8 -*-
import pdf_to_json_to_word as toword
import os
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-DocTranslation")

input_path = "mypaddle/upgit/document_sample.pdf"
output_path = "mypaddle/upgit/output"

#  该部分不需要用到翻译模块，故去掉翻译部分

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
    # 收集每页解析到的json数据
    json_list = []

    for res in visual_predict_res:
        layout_parsing_result = res["layout_parsing_result"]
        # 每页的json
        json_data = layout_parsing_result._to_json()
        json_list.append(json_data)
        
        layout_parsing_result.markdown
        layout_parsing_result.save_to_markdown(output_path)
    
# 将所有的json中的block抽取出来，合并成一个只含block的json
merged_json_path = toword.merge_block(json_list,output_path=output_path)

# 调用转换成 Word 的函数
base_name = os.path.splitext(os.path.basename(input_path))[0]

toword.blocks_to_word(
    json_path=merged_json_path,
    word_output_path=f"{output_path}/{base_name}_toword.docx",
    image_base_path=f"{output_path}/imgs",
    input_path=input_path,
    output_path=output_path,
)