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

from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="PP-ChatOCRv4-doc")

img_path = "./test_samples/研报2_11.jpg"
key_list = ["三位一体养老生态系统包含哪些"]

# img_path = "./test_samples/财报1.pdf"
# key_list = ['公司全称是什么']


def load_mllm_results():
    """load mllm results"""
    import json

    predict_file_path = "/paddle/icode/baidu/paddlex_closed/evaluation/pipelines/ppchatocr/backend_predict_files/predict_mix_doc_v1_2B-1209.json"
    mllm_predict_dict = {}
    with open(predict_file_path, "r") as fin:
        predict_infos_list = json.load(fin)
        for predict_infos in predict_infos_list:
            img_name = predict_infos["image_path"]
            predict_info_list = predict_infos["predict_info_list"]
            for predict_info in predict_info_list:
                key = img_name + "_" + predict_info["question"]
                mllm_predict_dict[key] = predict_info
    return mllm_predict_dict


mllm_predict_dict_all = load_mllm_results()

visual_predict_res = pipeline.visual_predict(
    img_path,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]
    print(layout_parsing_result)
    layout_parsing_result.print()
    layout_parsing_result.save_to_img("./output")
    layout_parsing_result.save_to_json("./output")
    layout_parsing_result.save_to_xlsx("./output")
    layout_parsing_result.save_to_html("./output")

pipeline.save_visual_info_list(
    visual_info_list, "./res_visual_info/tmp_visual_info.json"
)

visual_info_list = pipeline.load_visual_info_list(
    "./res_visual_info/tmp_visual_info.json"
)

vector_info = pipeline.build_vector(visual_info_list, flag_save_bytes_vector=True)

pipeline.save_vector(vector_info, "./res_visual_info/tmp_vector_info.json")

vector_info = pipeline.load_vector("./res_visual_info/tmp_vector_info.json")

mllm_predict_dict = {}
image_name = img_path.split("/")[-1]
for key in key_list:
    mllm_predict_key = image_name + "_" + key
    mllm_result = ""
    if mllm_predict_key in mllm_predict_dict_all:
        mllm_result = mllm_predict_dict_all[mllm_predict_key]["predicts"]
    mllm_predict_dict[key] = mllm_result

chat_result = pipeline.chat(
    key_list,
    visual_info_list,
    vector_info=vector_info,
    mllm_predict_dict=mllm_predict_dict,
)

print(chat_result)
