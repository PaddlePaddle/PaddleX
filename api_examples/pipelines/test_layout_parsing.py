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

pipeline = create_pipeline(pipeline="layout_parsing")

output = pipeline.predict(
    "./test_samples/demo_paper.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

# output = pipeline.predict(
#     "./test_samples/layout.jpg",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_common_ocr=True,
#     use_seal_recognition=True,
#     use_table_recognition=True,
# )

# output = pipeline.predict(
#     "./test_samples/test_layout_parsing.jpg",
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True,
#     use_common_ocr=True,
#     use_seal_recognition=True,
#     use_table_recognition=True,
# )

# output = pipeline.predict(
#     "./test_samples/财报1.pdf",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_common_ocr=True,
#     use_seal_recognition=True,
#     use_table_recognition=True,
# )

# output = pipeline.predict(
#     "./test_samples/layout_double_column.png",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_common_ocr=True,
#     use_seal_recognition=True,
#     use_table_recognition=True,
# )

for res in output:
    res.print()
    res.save_to_img("./output")
    res.save_to_json("./output")
    res.save_to_xlsx("./output")
    res.save_to_html("./output")
