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

pipeline = create_pipeline(pipeline="doc_preprocessor")

output = pipeline.predict(
    "./test_samples/img_rot180_demo.jpg",
    use_doc_orientation_classify=True,
    use_doc_unwarping=False,
)

# output = pipeline.predict(
#     "./test_samples/img_rot180_demo.jpg",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=True,
# )

# output = pipeline.predict(
#     "./test_samples/img_rot180_demo.jpg",
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True,
# )

# output = pipeline.predict(
#     "./test_samples/img_rot180_demo.jpg",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
# )

# output = pipeline.predict(
#     "./test_samples/doc_distort_test.jpg",
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True
# )

for res in output:
    print(res)
    res.print()
    res.save_to_img("./output")
    res.save_to_json("./output")
