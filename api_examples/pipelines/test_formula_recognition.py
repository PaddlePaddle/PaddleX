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

pipeline = create_pipeline(pipeline="formula_recognition")

output = pipeline.predict(
    "./test_samples/general_formula_recognition01.png", use_layout_detection=True
)

# output = pipeline.predict(
#     "./test_samples/general_formula_recognition01.pdf",
#     use_layout_detection=True,
# )

# output = pipeline.predict(
#     "./test_samples/general_formula_recognition02.png",
#     use_layout_detection=False,
# )

# img_list = [ "./test_samples/general_formula_recognition03.png", \
#     "./test_samples/general_formula_recognition04.png", \
#         "./test_samples/general_formula_recognition05.png",]
# output = pipeline.predict(
#     img_list,
#     use_layout_detection=True,
# )

for res in output:
    print(res)
    res.print()
    res.save_to_img("./output")
    res.save_to_json("./output")
