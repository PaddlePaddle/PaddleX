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

pipeline = create_pipeline(pipeline="OCR")

output = pipeline.predict(
    "./test_samples/general_ocr_002.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# output = pipeline.predict(
#     "./test_samples/general_ocr_002.png",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     text_rec_score_thresh = 0.5
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_002.png",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
#     text_det_unclip_ratio=3.0,
#     text_det_limit_side_len=1920
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_002.png",
#     use_doc_orientation_classify=True,
#     use_doc_unwarping=True,
#     use_textline_orientation=False
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_003.jpg",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_003.jpg",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=True
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_002_rotate_90.png",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_002_rotate_90.png",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=True
# )

# output = pipeline.predict(
#     "./test_samples/general_ocr_002.png")

# output = pipeline.predict("./test_samples/财报1.pdf")

for res in output:
    print(res)
    res.print()
    res.save_to_img("./output")
    res.save_to_json("./output")
