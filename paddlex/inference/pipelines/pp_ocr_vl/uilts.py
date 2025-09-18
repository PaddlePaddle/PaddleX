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
from typing import Any, Dict, List,Tuple

import numpy as np
import cv2
from PIL import Image
from collections import Counter
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
                overlap_ratio > 0.7
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


# def merge_images(images):
#     """
#     Merge a list of images (np.array) into a single image (PIL.Image).
#     """
#     if not images:
#         return None

#     # Calculate total height and max width
#     total_height = sum(
#         image.shape[0] for image in images
#     )  # image.shape[0] is the height
#     max_width = max(image.shape[1] for image in images)  # image.shape[1] is the width

#     # Create a new blank image with white background
#     new_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

#     current_height = 0
#     for image in images:
#         pil_image = Image.fromarray(image)  # Convert np.array to PIL.Image
#         x_offset = (max_width - pil_image.width) // 2
#         new_image.paste(pil_image, (x_offset, current_height))
#         current_height += pil_image.height

#     return np.array(new_image)

# def merge_blocks(blocks, non_merge_labels):
#     current_group_images = []
#     group_index = 0  # 当前合并组起始下标

#     for i, block in enumerate(blocks):
#         block_img = block["img"]
#         block_bbox = block["box"]
#         block_label = block["label"]

#         # non_merge_labels 内的直接跳过，不做合并
#         if block_label in non_merge_labels:
#             if current_group_images:
#                 merged_image = merge_images(current_group_images)
#                 for j in range(group_index, i):
#                     if j == group_index:
#                         blocks[j]["img"] = merged_image
#                     else:
#                         blocks[j]["img"] = None
#                 current_group_images = []
#             # 非合并块自己保留
#             blocks[i]["img"] = block_img
#             group_index = i + 1
#             continue

#         # 第一个可合并块，启动新group
#         if not current_group_images:
#             current_group_images = [block_img]
#             group_index = i
#             continue

#         # cross判断逻辑
#         prev_block = blocks[i - 1]
#         prev_bbox = prev_block["box"]
#         prev_label = prev_block["label"]

#         # 只合并cross：无水平投影重叠 + 下一个block在右侧 + label相同
#         iou = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
#         is_cross = (
#             iou == 0
#             and block_label == prev_label
#             and block_bbox[0] > prev_bbox[2]  # 当前左边界大于前一个右边界
#         )

#         if is_cross:
#             current_group_images.append(block_img)
#         else:
#             # 只在 cross 合并，其他情况直接分组，当前block自成一组
#             if len(current_group_images) > 1:
#                 merged_image = merge_images(current_group_images)
#                 for j in range(group_index, i):
#                     if j == group_index:
#                         blocks[j]["img"] = merged_image
#                     else:
#                         blocks[j]["img"] = None
#             else:
#                 # 只有一个，不需要合并
#                 blocks[group_index]["img"] = current_group_images[0]

#             group_index = i
#             current_group_images = [block_img]

#     # 处理最后一组
#     if current_group_images:
#         if len(current_group_images) > 1:
#             merged_image = merge_images(current_group_images)
#             for j in range(group_index, len(blocks)):
#                 if j == group_index:
#                     blocks[j]["img"] = merged_image
#                 else:
#                     blocks[j]["img"] = None
#         else:
#             blocks[group_index]["img"] = current_group_images[0]

#     return blocks


def to_pil_image(img):
    return img if isinstance(img, Image.Image) else Image.fromarray(img)


def to_np_array(img):
    return np.array(img) if isinstance(img, Image.Image) else img


def calc_merged_wh(images):
    widths = [to_pil_image(img).width for img in images]
    heights = [to_pil_image(img).height for img in images]
    w = max(widths)
    h = sum(heights)
    return w, h


def merge_images(images, aligns="center"):
    """
    Merge a list of images (np.array or PIL.Image) into a single image (np.array).
    aligns: 单个字符串或list，比如["left", "center"]，每步指定对齐方式。
    """
    if not images:
        return None
    if len(images) == 1:
        return to_np_array(images[0])
    # aligns参数标准化
    if isinstance(aligns, str):
        aligns = [aligns] * (len(images) - 1)
    if len(aligns) != len(images) - 1:
        raise ValueError("aligns长度需等于images数量减一")
    merged = to_pil_image(images[0])
    for i in range(1, len(images)):
        img2 = to_pil_image(images[i])
        align = aligns[i - 1]
        w = max(merged.width, img2.width)
        h = merged.height + img2.height
        new_img = Image.new("RGB", (w, h), (255, 255, 255))
        if align == "center":
            x1 = (w - merged.width) // 2
            x2 = (w - img2.width) // 2
        elif align == "right":
            x1 = w - merged.width
            x2 = w - img2.width
        else:  # left
            x1 = x2 = 0
        new_img.paste(merged, (x1, 0))
        new_img.paste(img2, (x2, merged.height))
        merged = new_img
    return to_np_array(merged)


def merge_blocks(blocks, non_merge_labels):
    blocks_to_merge = []
    non_merge_blocks = {}
    for idx, block in enumerate(blocks):
        if block["label"] in non_merge_labels:
            non_merge_blocks[idx] = block
        else:
            blocks_to_merge.append((idx, block))

    merged_groups = []
    current_group = []
    current_indices = []
    current_aligns = []

    def is_aligned(a1, a2):
        return abs(a1 - a2) <= 5

    def get_alignment(block_bbox, prev_bbox):
        if is_aligned(block_bbox[0], prev_bbox[0]):
            return "left"
        elif is_aligned(block_bbox[2], prev_bbox[2]):
            return "right"
        else:
            return "center"

    def overlapwith_other_box(block_idx, prev_idx, blocks):
        prev_bbox = blocks[prev_idx]["box"]
        block_bbox = blocks[block_idx]["box"]
        x1 = min(prev_bbox[0], block_bbox[0])
        y1 = min(prev_bbox[1], block_bbox[1])
        x2 = max(prev_bbox[2], block_bbox[2])
        y2 = max(prev_bbox[3], block_bbox[3])
        min_box = [x1, y1, x2, y2]
        for idx, other_block in enumerate(blocks):
            if idx in [block_idx, prev_idx]:
                continue
            other_bbox = other_block["box"]
            if calculate_overlap_ratio(min_box, other_bbox) > 0:
                return True
        return False

    for i, (idx, block) in enumerate(blocks_to_merge):
        if not current_group:
            current_group = [block]
            current_indices = [idx]
            current_aligns = []
            continue

        prev_idx, prev_block = blocks_to_merge[i - 1]
        prev_bbox = prev_block["box"]
        prev_label = prev_block["label"]
        block_bbox = block["box"]
        block_label = block["label"]

        iou_h = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
        is_cross = (
            iou_h == 0
            and block_label == "text"
            and block_label == prev_label
            and block_bbox[0] > prev_bbox[2]
            and block_bbox[1] < prev_bbox[3]
            and block_bbox[0] - prev_bbox[2]
            < max(prev_bbox[2] - prev_bbox[0], block_bbox[2] - block_bbox[0]) * 0.3
        )
        is_updown_align = (
            iou_h > 0
            and block_label in ["text"]
            and block_label == prev_label
            and block_bbox[3] >= prev_bbox[1]
            and abs(block_bbox[1]-prev_bbox[3]) < max(prev_bbox[3] - prev_bbox[1], block_bbox[3] - block_bbox[1]) * 0.5
            and (
                is_aligned(block_bbox[0], prev_bbox[0])
                ^ is_aligned(block_bbox[2], prev_bbox[2])
            )
            and overlapwith_other_box(idx, prev_idx, blocks)
        )
        if is_cross:
            align_mode = "center"
        elif is_updown_align:
            align_mode = get_alignment(block_bbox, prev_bbox)
        else:
            align_mode = None

        if is_cross or is_updown_align:
            current_group.append(block)
            current_indices.append(idx)
            current_aligns.append(align_mode)
        else:
            merged_groups.append((current_indices, current_group, current_aligns))
            current_group = [block]
            current_indices = [idx]
            current_aligns = []
    if current_group:
        merged_groups.append((current_indices, current_group, current_aligns))

    group_ranges = []
    for group_indices, group, aligns in merged_groups:
        start, end = min(group_indices), max(group_indices)
        group_ranges.append((start, end, group_indices, aligns))

    result_blocks = []
    used_indices = set()
    idx = 0
    while idx < len(blocks):
        group_found = False
        for (start, end, group_indices, aligns), (g_indices, g_blocks, g_aligns) in zip(
            group_ranges, merged_groups
        ):
            if idx == start and all(i not in used_indices for i in group_indices):
                group_found = True
                imgs = [blocks[i]["img"] for i in group_indices]
                merge_aligns = aligns if aligns else []
                w, h = calc_merged_wh(imgs)
                if h == 0 or w == 0:
                    aspect_ratio = float("inf")
                else:
                    aspect_ratio = h / w
                if aspect_ratio >= 3:
                    # 不合并，分别处理
                    for j, block_idx in enumerate(group_indices):
                        block = blocks[block_idx].copy()
                        block["img"] = blocks[block_idx]["img"]
                        block["merge_aligns"] = None
                        result_blocks.append(block)
                        used_indices.add(block_idx)
                else:
                    merged_img = merge_images(imgs, merge_aligns)
                    for j, block_idx in enumerate(group_indices):
                        block = blocks[block_idx].copy()
                        block["img"] = merged_img if j == 0 else None
                        block["merge_aligns"] = merge_aligns if j == 0 else None
                        result_blocks.append(block)
                        used_indices.add(block_idx)
                # 插入组内 non_merge 块
                insert_list = []
                for n_idx in range(start + 1, end):
                    if n_idx in non_merge_blocks:
                        insert_list.append(n_idx)
                for n_idx in insert_list:
                    result_blocks.append(non_merge_blocks[n_idx])
                    used_indices.add(n_idx)
                idx = end + 1
                break
        if group_found:
            continue
        if idx in non_merge_blocks and idx not in used_indices:
            result_blocks.append(non_merge_blocks[idx])
            used_indices.add(idx)
        idx += 1

    return result_blocks


def paint_token(image, box, token_str):
    """
    image: numpy.ndarray, 图像
    box: (x1, y1, x2, y2), 填充的矩形区域
    token: str, 要写入的内容
    返回: 修改后的图像
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    img = image.copy()
    # 填充白色
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,255,255), thickness=-1)

    # 计算区域宽高
    box_w = x2 - x1
    box_h = y2 - y1

    # 自动调整字体大小，使文本不会超出box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    # 先尝试较大的字体，再逐步减小
    while font_scale > 0:
        (text_w, text_h), baseline = cv2.getTextSize(token_str, font, font_scale, font_thickness)
        if text_w <= box_w * 0.9 and text_h + baseline <= box_h * 0.9:
            break
        font_scale -= 0.1
    if font_scale <= 0:  # 还是放不下，缩小到最小
        font_scale = 0.2
        (text_w, text_h), baseline = cv2.getTextSize(token_str, font, font_scale, font_thickness)

    # 计算文本左下角坐标，使其居中
    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h + text_h) // 2

    # 画文本
    cv2.putText(img, token_str, (text_x, text_y), font, font_scale, (0,0,0), font_thickness, lineType=cv2.LINE_AA)

    return img


def tokenize_figure_of_table(table_block_img, table_box, figures):
    token_map = {}
    table_x_min, table_y_min, table_x_max, table_y_max = table_box
    for figure_id, figure in enumerate(figures):
        figure_x_min, figure_y_min, figure_x_max, figure_y_max = figure["coordinate"]
        if figure_x_min >= table_x_min and figure_y_min >= table_y_min and figure_x_max <= table_x_max and figure_y_max <= table_y_max:
            draw_box = [figure_x_min - table_x_min, figure_y_min - table_y_min, figure_x_max - table_x_min, figure_y_max - table_y_min]
            token_str = "[F" + str(figure_id) + "]"
            table_block_img = paint_token(table_block_img, draw_box, token_str)
            token_map[token_str] = f'<img src="{figure["path"]}" >'
    return table_block_img, token_map


def untokenize_figure_of_table(table_res_str, figure_token_map):
    def repl(match):
        token_id = match.group(1)
        token = f"[F{token_id}]"
        return figure_token_map.get(token, match.group(0))
    pattern = r'\[F(\d+)\]'
    return re.sub(pattern, repl, table_res_str)


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

NON_CAPTURING_TAG_GROUP = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
OTSL_FIND_PATTERN = re.compile(
    f"{NON_CAPTURING_TAG_GROUP}.*?(?={NON_CAPTURING_TAG_GROUP}|$)",flags=re.DOTALL
)


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


def otsl_pad_to_sqr_v2(otsl_str: str) -> str:

    assert isinstance(otsl_str, str)

    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        # NOTE 直接当单行表格处理
        return otsl_str + OTSL_NL

    lines = otsl_str.split(OTSL_NL)

    row_data = []
    for line in lines:
        if not line:
            continue

        # NOTE 拆成单元格表达形式
        raw_cells = OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue

        total_len = len(raw_cells)  # NOTE 当前行的整体单元格数量
        # NOTE 需要计算出该行允许的最小单元格数量
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1

        row_data.append(
            {"raw_cells": raw_cells, "total_len": total_len, "min_len": min_len}
        )

    if not row_data:
        return OTSL_NL

    global_min_width = max(row["min_len"] for row in row_data) if row_data else 0
    max_total_len = max(row["total_len"] for row in row_data) if row_data else 0

    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)

    min_total_cost = float("inf")
    optimal_width = search_end  # NOTE 默认需要补充到最大长度

    for width in range(search_start, search_end + 1):
        current_total_cost = sum(abs(row["total_len"] - width) for row in row_data)

        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            optimal_width = width

    # NOTE 基于 optimal_width 重建表格
    repaired_lines = []
    for row in row_data:
        cells = row["raw_cells"]
        current_len = len(cells)

        if current_len > optimal_width:  # NOTE 末尾安全截断
            new_cells = cells[:optimal_width]
        else:  # NOTE 补充
            padding = [OTSL_ECEL] * (optimal_width - current_len)
            new_cells = cells + padding

        repaired_lines.append("".join(new_cells))

    return OTSL_NL.join(repaired_lines) + OTSL_NL


def convert_otsl_to_html(otsl_content: str):
    """NOTE otsl v1.0转换成html，只能有6个tag: <fcel>, <ecel>, <nl>, <lcel>, <ucel>, <xcel>

    注意点：
        1. <fcel>之后一定有内容，ecel之后一定没内容，否则会引入乱码
    """
    otsl_content = otsl_pad_to_sqr_v2(otsl_content)
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)

    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )

    return export_to_html(table_data)


def find_shortest_repeating_substring(s: str) -> str | None:
    """
    Finds the shortest repeating substring that constitutes the ENTIRE string s.
    e.g., s='abcabcabc' returns 'abc'. s='abab' returns 'ab'. s='abca' returns None.
    """
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                return substring
    return None
 
# --- NEW FUNCTION: Detects repeating phrases at the end of a string ---
def find_repeating_suffix(s: str, min_len: int = 8, min_repeats: int = 5) -> Tuple[str, str, int] | None:
    """
    Finds if a string ends with a repeating phrase.
    e.g., s='start...phrase,phrase,phrase,' returns ('start...', 'phrase,', 3)
    
    Args:
        s (str): The input string.
        min_len (int): The minimum length of the repeating unit to consider.
        min_repeats (int): The minimum number of repetitions to trigger truncation.
 
    Returns:
        A tuple (prefix, unit, count) if a repeating suffix is found, otherwise None.
    """
    # Iterate through possible lengths of the repeating unit, from longest to shortest.
    for i in range(len(s) // (min_repeats), min_len - 1, -1):
        unit = s[-i:]
        
        # Quick check: does the string end with the unit repeated at least min_repeats times?
        if s.endswith(unit * min_repeats):
            # If so, find the exact number of repetitions
            count = 0
            temp_s = s
            while temp_s.endswith(unit):
                temp_s = temp_s[:-i]
                count += 1
            
            # Return the non-repeating prefix, the unit, and its count
            start_index = len(s) - (count * i)
            return s[:start_index], unit, count
    return None

def truncate_repetitive_content(content: str, line_threshold: int = 10, char_threshold: int = 10, min_len: int = 10) -> (str, str):
    """
    Intelligently detects and truncates character, phrase, or line-level repetitive content.
    This version uses a more aggressive strategy for suffix repetition: it deletes the entire repeating part.
    """
    stripped_content = content.strip()
    if not stripped_content:
        return content, ""
 
    # --- MODIFIED LOGIC with AGGRESSIVE DELETION ---
    # Priority 1: Check for phrase-level suffix repetition in single, long lines.
    if '\n' not in stripped_content and len(stripped_content) > 100:
        suffix_match = find_repeating_suffix(stripped_content, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, repeating_unit, count = suffix_match
            # Ensure the repeating part is a significant portion of the whole string
            if len(repeating_unit) * count > len(stripped_content) * 0.5:
                # The log message is updated to reflect the new action
                truncated_info = f"[截断信息: 检测到单行内短语重复，'{repeating_unit}' 在末尾连续出现 {count} 次，已将重复部分完全删除。]"
                # Return ONLY the non-repeating prefix
                return prefix, truncated_info
    # --- END of MODIFIED LOGIC ---
 
    # Priority 2: Check for full-string character-level repetition (e.g., 'ababab')
    # For this type, keeping one unit is still reasonable (e.g., '----' -> '-')
    if '\n' not in stripped_content and len(stripped_content) > min_len:
        repeating_unit = find_shortest_repeating_substring(stripped_content)
        if repeating_unit:
            count = len(stripped_content) // len(repeating_unit)
            if count >= char_threshold:
                truncated_info = f"[截断信息: 检测到字符级重复，'{repeating_unit}' 共出现 {count} 次，已合并为一次。]"
                return repeating_unit, truncated_info
 
    # Priority 3: Check for line-level repetition (e.g., the same line repeated many times)
    # For this type as well, keeping one line is often the desired behavior
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    if not lines:
        return content, ""
 
    total_lines = len(lines)
    if total_lines < line_threshold:
        return content, ""
        
    line_counts = Counter(lines)
    most_common_line, count = line_counts.most_common(1)[0]
    
    if count >= line_threshold and (count / total_lines) >= 0.8:
        truncated_info = f"[截断信息: 检测到行级重复，'{most_common_line}' 共出现 {count} 次，已合并为一次。]"
        return most_common_line, truncated_info
    
    return content, ""
