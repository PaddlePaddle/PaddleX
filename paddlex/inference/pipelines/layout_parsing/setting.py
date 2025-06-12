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


XYCUT_SETTINGS = {
    "child_block_overlap_ratio_threshold": 0.1,
    "edge_distance_compare_tolerance_len": 2,
    "distance_weight_map": {
        "edge_weight": 10**4,
        "up_edge_weight": 1,
        "down_edge_weight": 0.0001,
    },
    "cross_layout_ref_text_block_words_num_threshold": 10,
}

REGION_SETTINGS = {
    "match_block_overlap_ratio_threshold": 0.6,
    "split_block_overlap_ratio_threshold": 0.4,
}

BLOCK_SETTINGS = {
    "title_conversion_area_ratio_threshold": 0.3,  # update paragraph_title -> doc_title
}

LINE_SETTINGS = {
    "line_height_iou_threshold": 0.6,  # For line segmentation of OCR results
    "delimiter_map": {
        "doc_title": " ",
        "content": "\n",
    },
}

BLOCK_LABEL_MAP = {
    "doc_title_labels": ["doc_title"],  # 文档标题
    "paragraph_title_labels": [
        "paragraph_title",
        "abstract_title",
        "reference_title",
        "content_title",
    ],  # 段落标题
    "vision_labels": [
        "image",
        "table",
        "chart",
        "flowchart",
        "figure",
    ],  # 图、表、印章、图表、图
    "vision_title_labels": [
        "table_title",
        "chart_title",
        "figure_title",
        "figure_table_chart_title",
    ],  # 图表标题
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
        "abstract_title",
        "refer_title",
        "content_title",
    ],
    "image_labels": ["image", "figure"],
}
