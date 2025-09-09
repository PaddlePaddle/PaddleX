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
import html
import itertools
import re
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from pydantic import BaseModel, computed_field, model_validator

from ..layout_parsing.utils import (
    calculate_bbox_area,
    calculate_overlap_ratio,
    calculate_projection_overlap_ratio,
)


def filter_overlap_boxes(
    layout_det_res: Dict[str, List[Dict]]
) -> Dict[str, List[Dict]]:
    """
    Filter out overlapping boxes from layout detection results based on overlap ratio.

    Args:
        layout_det_res (Dict[str, List[Dict]]): Dictionary containing detection results with 'boxes' key.

    Returns:
        Dict[str, List[Dict]]: Filtered layout detection results with overlapping boxes removed.
    """
    layout_det_res_filted = deepcopy(layout_det_res)
    boxes = [
        box for box in layout_det_res_filted["boxes"] if box["label"] != "reference"
    ]
    dropped_indexes = set()

    # Iterate over each pair of boxes to find overlaps
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            # Skip boxes that are already marked for removal
            if i in dropped_indexes or j in dropped_indexes:
                continue

            # Calculate the overlap ratio
            overlap_ratio = calculate_overlap_ratio(
                boxes[i]["coordinate"], boxes[j]["coordinate"], "small"
            )

            # If overlap ratio is significant, mark one of the boxes for removal
            if (
                overlap_ratio > 0.9
            ):  # Assuming 1 is the threshold for significant overlap
                # Here we are assuming higher score is preferable, you might want to adjust this logic
                box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                if (
                    boxes[i]["label"] == "image" or boxes[j]["label"] == "image"
                ) and boxes[i]["label"] != boxes[j]["label"]:
                    continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)

    # Remove marked boxes
    layout_det_res_filted["boxes"] = [
        box for idx, box in enumerate(boxes) if idx not in dropped_indexes
    ]

    return layout_det_res_filted


def merge_images(images):
    """
    Merge a list of images (np.array) into a single image (PIL.Image).
    """
    if not images:
        return None

    # Calculate total height and max width
    total_height = sum(
        image.shape[0] for image in images
    )  # image.shape[0] is the height
    max_width = max(image.shape[1] for image in images)  # image.shape[1] is the width

    # Create a new blank image with white background
    new_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

    current_height = 0
    for image in images:
        pil_image = Image.fromarray(image)  # Convert np.array to PIL.Image
        x_offset = (max_width - pil_image.width) // 2
        new_image.paste(pil_image, (x_offset, current_height))
        current_height += pil_image.height

    return np.array(new_image)


# def merge_blocks(blocks, non_merge_labels):
#     current_group_images = []
#     current_group_label = None
#     crossing = False
#     group_index = 0

#     for i, block in enumerate(blocks):
#         block_img = block["img"]
#         block_bbox = block["box"]
#         block_label = block["label"]

#         if block_label in non_merge_labels:
#             # If the current block's label is in the non-merge list, reset grouping
#             if current_group_images:
#                 merged_image = merge_images(current_group_images)
#                 for j in range(group_index, i):
#                     if j == group_index:
#                         blocks[j]["img"] = merged_image
#                     else:
#                         blocks[j]["img"] = None
#             # Reset for the non-merge block
#             blocks[i]["img"] = block_img
#             current_group_images = []
#             current_group_label = None
#             crossing = False
#             group_index = i + 1
#             continue

#         if not current_group_images:
#             current_group_images = [block_img]
#             current_group_label = block_label
#             crossing = False
#             continue

#         prev_block = blocks[i - 1]
#         iou = calculate_projection_overlap_ratio(
#             block_bbox, prev_block["box"], "horizontal"
#         )

#         if iou == 0 and block_label == current_group_label:
#             current_group_images.append(block_img)
#             crossing = True
#         else:
#             if crossing:
#                 merged_image = merge_images(current_group_images)
#                 for j in range(group_index, i):
#                     if j == group_index:
#                         blocks[j]["img"] = merged_image
#                     else:
#                         blocks[j]["img"] = None
#                 group_index = i
#                 current_group_images = [block_img]
#                 current_group_label = block_label
#                 crossing = False
#             else:
#                 if iou > 0 and block_label == current_group_label:
#                     current_group_images.append(block_img)
#                 else:
#                     merged_image = merge_images(current_group_images)
#                     for j in range(group_index, i):
#                         if j == group_index:
#                             blocks[j]["img"] = merged_image
#                         else:
#                             blocks[j]["img"] = None
#                     group_index = i
#                     current_group_images = [block_img]
#                     current_group_label = block_label
#                     crossing = False

#     if current_group_images:
#         merged_image = merge_images(current_group_images)
#         for j in range(group_index, len(blocks)):
#             if j == group_index:
#                 blocks[j]["img"] = merged_image
#             else:
#                 blocks[j]["img"] = None

#     return blocks


def merge_blocks(blocks, non_merge_labels):
    current_group_images = []
    group_index = 0  # 当前合并组起始下标

    for i, block in enumerate(blocks):
        block_img = block["img"]
        block_bbox = block["box"]
        block_label = block["label"]

        # non_merge_labels 内的直接跳过，不做合并
        if block_label in non_merge_labels:
            if current_group_images:
                merged_image = merge_images(current_group_images)
                for j in range(group_index, i):
                    if j == group_index:
                        blocks[j]["img"] = merged_image
                    else:
                        blocks[j]["img"] = None
                current_group_images = []
            # 非合并块自己保留
            blocks[i]["img"] = block_img
            group_index = i + 1
            continue

        # 第一个可合并块，启动新group
        if not current_group_images:
            current_group_images = [block_img]
            group_index = i
            continue

        # cross判断逻辑
        prev_block = blocks[i - 1]
        prev_bbox = prev_block["box"]
        prev_label = prev_block["label"]

        # 只合并cross：无水平投影重叠 + 下一个block在右侧 + label相同
        iou = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
        is_cross = (
            iou == 0
            and block_label == prev_label
            and block_bbox[0] > prev_bbox[2]  # 当前左边界大于前一个右边界
        )

        if is_cross:
            current_group_images.append(block_img)
        else:
            # 只在 cross 合并，其他情况直接分组，当前block自成一组
            if len(current_group_images) > 1:
                merged_image = merge_images(current_group_images)
                for j in range(group_index, i):
                    if j == group_index:
                        blocks[j]["img"] = merged_image
                    else:
                        blocks[j]["img"] = None
            else:
                # 只有一个，不需要合并
                blocks[group_index]["img"] = current_group_images[0]

            group_index = i
            current_group_images = [block_img]

    # 处理最后一组
    if current_group_images:
        if len(current_group_images) > 1:
            merged_image = merge_images(current_group_images)
            for j in range(group_index, len(blocks)):
                if j == group_index:
                    blocks[j]["img"] = merged_image
                else:
                    blocks[j]["img"] = None
        else:
            blocks[group_index]["img"] = current_group_images[0]

    return blocks


# def merge_images(images, aligns="center"):
#     """
#     Merge a list of images (np.array or PIL.Image) into a single image (np.array).
#     aligns: 单个字符串或list，比如["left", "center"]，每步指定对齐方式。
#     """
#     if not images:
#         return None
#     if len(images) == 1:
#         # 只一张图，直接返回array
#         if isinstance(images[0], Image.Image):
#             return np.array(images[0])
#         return images[0]

#     # 兼容aligns为字符串的情况
#     if isinstance(aligns, str):
#         aligns = [aligns] * (len(images) - 1)
#     if len(aligns) != len(images) - 1:
#         raise ValueError("aligns长度需等于images数量减一")

#     # 逐步两两合并
#     merged = images[0]
#     for i in range(1, len(images)):
#         align = aligns[i-1]
#         merged = merge_two_images(merged, images[i], align)
#     return merged

# def merge_two_images(img1, img2, align="center"):
#     """
#     合并两张图片，img1在上img2在下，按宽度对齐
#     支持 img1, img2 为 np.array 或 PIL.Image
#     返回 np.array
#     """
#     # 转成PIL
#     if not isinstance(img1, Image.Image):
#         img1 = Image.fromarray(img1)
#     if not isinstance(img2, Image.Image):
#         img2 = Image.fromarray(img2)
#     w = max(img1.width, img2.width)
#     h = img1.height + img2.height
#     new_image = Image.new("RGB", (w, h), (255, 255, 255))
#     # 上图/下图对齐方式
#     if align == "center":
#         x1 = (w - img1.width) // 2
#         x2 = (w - img2.width) // 2
#     elif align == "right":
#         x1 = w - img1.width
#         x2 = w - img2.width
#     else:  # left
#         x1 = 0
#         x2 = 0
#     new_image.paste(img1, (x1, 0))
#     new_image.paste(img2, (x2, img1.height))
#     return np.array(new_image)

# # 下面是merge_blocks的重构
# def merge_blocks(blocks, non_merge_labels):
#     blocks_to_merge = []
#     non_merge_blocks = {}
#     for idx, block in enumerate(blocks):
#         if block["label"] in non_merge_labels:
#             non_merge_blocks[idx] = block
#         else:
#             blocks_to_merge.append((idx, block))

#     merged_groups = []
#     current_group = []
#     current_indices = []
#     current_aligns = []

#     def is_aligned(a1, a2):
#         return abs(a1 - a2) <= 5

#     def get_alignment(block_bbox, prev_bbox):
#         if is_aligned(block_bbox[0], prev_bbox[0]):
#             return "left"
#         elif is_aligned(block_bbox[2], prev_bbox[2]):
#             return "right"
#         else:
#             return "center"

#     for i, (idx, block) in enumerate(blocks_to_merge):
#         if not current_group:
#             current_group = [block]
#             current_indices = [idx]
#             current_aligns = []
#             continue

#         prev_idx, prev_block = blocks_to_merge[i-1]
#         prev_bbox = prev_block["box"]
#         prev_label = prev_block["label"]
#         block_bbox = block["box"]
#         block_label = block["label"]

#         iou_h = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
#         is_cross = (
#             iou_h == 0
#             and block_label == prev_label
#             and block_bbox[0] > prev_bbox[2]
#         )
#         is_updown_align = (
#             iou_h > 0
#             and block_label == prev_label
#             and (
#                 block_bbox[1] >= prev_bbox[3] or prev_bbox[1] >= block_bbox[3]
#             )
#             and (is_aligned(block_bbox[0], prev_bbox[0]) ^ is_aligned(block_bbox[2], prev_bbox[2]))
#         )
#         if is_cross:
#             align_mode = "center"
#         elif is_updown_align:
#             align_mode = get_alignment(block_bbox, prev_bbox)
#         else:
#             align_mode = None

#         if is_cross or is_updown_align:
#             current_group.append(block)
#             current_indices.append(idx)
#             current_aligns.append(align_mode)  # 记录和前一个的合并方式
#         else:
#             # 一个组完结，保存组、索引、组内合并方式
#             merged_groups.append((current_indices, current_group, current_aligns))
#             current_group = [block]
#             current_indices = [idx]
#             current_aligns = []
#     if current_group:
#         merged_groups.append((current_indices, current_group, current_aligns))

#     group_ranges = []
#     for group_indices, group, aligns in merged_groups:
#         start, end = min(group_indices), max(group_indices)
#         group_ranges.append((start, end, group_indices, aligns))

#     result_blocks = []
#     used_indices = set()
#     idx = 0
#     while idx < len(blocks):
#         group_found = False
#         for (start, end, group_indices, aligns), (g_indices, g_blocks, g_aligns) in zip(group_ranges, merged_groups):
#             if idx == start and all(i not in used_indices for i in group_indices):
#                 group_found = True
#                 imgs = [blocks[i]["img"] for i in group_indices]
#                 # 组内多步合并，两两合并，每次用aligns中的方式
#                 merged_img = imgs[0]
#                 merge_aligns = aligns if aligns else []
#                 for j in range(1, len(imgs)):
#                     align = merge_aligns[j-1] if j-1 < len(merge_aligns) and merge_aligns[j-1] else "left"
#                     merged_img = merge_two_images(merged_img, imgs[j], align)
#                 # 填充结果
#                 for j, block_idx in enumerate(group_indices):
#                     block = blocks[block_idx].copy()
#                     block["img"] = merged_img if j == 0 else None
#                     block["merge_aligns"] = merge_aligns if j == 0 else None
#                     result_blocks.append(block)
#                     used_indices.add(block_idx)
#                 # 插入组内 non_merge 块
#                 insert_list = []
#                 for n_idx in range(start + 1, end):
#                     if n_idx in non_merge_blocks:
#                         insert_list.append(n_idx)
#                 for n_idx in insert_list:
#                     result_blocks.append(non_merge_blocks[n_idx])
#                     used_indices.add(n_idx)
#                 idx = end + 1
#                 break
#         if group_found:
#             continue
#         if idx in non_merge_blocks and idx not in used_indices:
#             result_blocks.append(non_merge_blocks[idx])
#             used_indices.add(idx)
#         idx += 1

#     return result_blocks


class TableCell(BaseModel):
    """Table
    Cell."""

    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """from_dict_format."""
        if isinstance(data, Dict):
            # Check if this is a native BoundingBox or a bbox from docling-ibm-models
            if (
                # "bbox" not in data
                # or data["bbox"] is None
                # or isinstance(data["bbox"], BoundingBox)
                "text"
                in data
            ):
                return data
            text = data["bbox"].get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "

                text = text.strip()
            data["text"] = text

        return data


class TableData(BaseModel):  # TBD
    """BaseTableData."""

    table_cells: List[TableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field
    @property
    def grid(
        self,
    ) -> List[List[TableCell]]:
        """grid."""
        # Initialise empty table data grid (only empty cells)
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]

        # Overwrite cells in table data for which there is actual cell content.
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell

        return table_data


"""
OTSL
"""
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"


def otsl_extract_tokens_and_text(s: str):
    # Pattern to match anything enclosed by < >
    # (including the angle brackets themselves)
    # pattern = r"(<[^>]+>)"
    pattern = (
        r"("
        + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL])
        + r")"
    )
    # Find all tokens (e.g. "<otsl>", "<loc_140>", etc.)
    tokens = re.findall(pattern, s)
    # Remove any tokens that start with "<loc_"
    tokens = [token for token in tokens]
    # Split the string by those tokens to get the in-between text
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts]
    # Remove any empty or purely whitespace strings from text_parts
    text_parts = [part for part in text_parts if part.strip()]

    return tokens, text_parts


def otsl_parse_texts(texts, tokens):
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # Check and complete the matrix
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)

        # Insert additional <ecel> to tags
        for row_idx, row in enumerate(split_row_tokens):
            while len(row) < max_cols:
                row.append(OTSL_ECEL)

        # Insert additional <ecel> to texts
        new_texts = []
        text_idx = 0

        for row_idx, row in enumerate(split_row_tokens):
            for col_idx, token in enumerate(row):
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in [
                        OTSL_NL,
                        OTSL_FCEL,
                        OTSL_ECEL,
                        OTSL_LCEL,
                        OTSL_UCEL,
                        OTSL_XCEL,
                    ]:
                        new_texts.append(texts[text_idx])
                        text_idx += 1

            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1

        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
        ]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2

            # Check next element(s) for lcel / ucel / xcel,
            # set properly row_span, col_span
            next_right_cell = ""
            if i + right_offset < len(texts):
                next_right_cell = texts[i + right_offset]

            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [
                OTSL_LCEL,
                OTSL_XCEL,
            ]:
                # we have horisontal spanning cell or 2d spanning cell
                col_span += count_right(
                    split_row_tokens,
                    c_idx + 1,
                    r_idx,
                    [OTSL_LCEL, OTSL_XCEL],
                )
            if next_bottom_cell in [
                OTSL_UCEL,
                OTSL_XCEL,
            ]:
                # we have a vertical spanning cell or 2d spanning cell
                row_span += count_down(
                    split_row_tokens,
                    c_idx,
                    r_idx + 1,
                    [OTSL_UCEL, OTSL_XCEL],
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [
            OTSL_FCEL,
            OTSL_ECEL,
            OTSL_LCEL,
            OTSL_UCEL,
            OTSL_XCEL,
        ]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def export_to_html(table_data: TableData):
    nrows = table_data.num_rows
    ncols = table_data.num_cols

    text = ""

    if len(table_data.table_cells) == 0:
        return ""

    body = ""

    grid = table_data.grid
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell: TableCell = grid[i][j]

            rowspan, rowstart = (
                cell.row_span,
                cell.start_row_offset_idx,
            )
            colspan, colstart = (
                cell.col_span,
                cell.start_col_offset_idx,
            )

            if rowstart != i:
                continue
            if colstart != j:
                continue

            content = html.escape(cell.text.strip())
            celltag = "td"
            if cell.column_header:
                celltag = "th"

            opening_tag = f"{celltag}"
            if rowspan > 1:
                opening_tag += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening_tag += f' colspan="{colspan}"'

            body += f"<{opening_tag}>{content}</{celltag}>"
        body += "</tr>"

    body = f"<table>{body}</table>"
    return body


def convert_otsl_to_html(otsl_content: str):
    """NOTE otsl v1.0转换成html，只能有6个tag: <fcel>, <ecel>, <nl>, <lcel>, <ucel>, <xcel>

    注意点：
        1. <fcel>之后一定有内容，ecel之后一定没内容，否则会引入乱码
    """
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )

    return export_to_html(table_data)
