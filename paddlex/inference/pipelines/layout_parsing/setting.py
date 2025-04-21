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

layout_order_config = {
    # 人工配置项
    "line_height_iou_threshold": 0.4,  # For line segmentation of OCR results
    "title_area_max_block_threshold": 0.3,  # update paragraph_title -> doc_title
    "block_label_match_iou_threshold": 0.1,
    "block_title_match_iou_threshold": 0.1,
    "doc_title_labels": ["doc_title"],  # 文档标题
    "paragraph_title_labels": ["paragraph_title"],  # 段落标题
    "vision_labels": [
        "image",
        "table",
        "chart",
        "figure",
    ],  # 图、表、印章、图表、图
    "vision_title_labels": ["table_title", "chart_title", "figure_title"],  # 图表标题
    "unordered_labels": [
        "aside_text",
        "seal",
        "number",
        "formula_number",
    ],
    "text_labels": ["text"],
    "header_labels": ["header", "header_image"],
    "footer_labels": ["footer", "footer_image", "footnote"],
    "visualize_index_labels": [
        "text",
        "formula",
        "algorithm",
        "reference",
        "content",
        "abstract",
        "paragraph_title",
        "doc_title",
        "table_title",
        "chart_title",
        "figure_title",
        "image",
        "table",
        "chart",
        "figure",
    ],
    # 自动补全配置项
    "layout_to_ocr_mapping": {},
    "all_layout_region_box": [],  # 区域box
    "doc_title_block_idxes": [],
    "paragraph_title_block_idxes": [],
    "text_title_labels": [],  # doc_title_labels+paragraph_title_labels
    "text_title_block_idxes": [],
    "vision_block_idxes": [],
    "vision_title_block_idxes": [],
    "vision_footnote_block_idxes": [],
    "text_block_idxes": [],
    "header_block_idxes": [],
    "footer_block_idxes": [],
    "unordered_block_idxes": [],
}
