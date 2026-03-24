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

"""Public formatting functions for document block conversion.

Extracted from:
  - paddlex/inference/pipelines/layout_parsing/result_v2.py
  - paddlex/inference/pipelines/paddleocr_vl/result.py
"""

from __future__ import annotations

import re
from functools import partial

# ---------------------------------------------------------------------------
# Title pattern (precompiled)
# ---------------------------------------------------------------------------


def compile_title_pattern():
    numbering_pattern = (
        r"(?:" + r"[1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?|" + r"[\(\（](?:[1-9][0-9]*|["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[\)\）]|" + r"["
        r"一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+"
        r"[、\.]?|" + r"(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)" + r")"
    )
    return re.compile(r"^\s*(" + numbering_pattern + r")(\s*)(.*)$")


TITLE_RE_PATTERN = compile_title_pattern()


# ---------------------------------------------------------------------------
# Formatting functions (block → str)
# ---------------------------------------------------------------------------


def format_title_func(block):
    """Normalize chapter title with '#' level indicator."""
    title = block.content
    match = TITLE_RE_PATTERN.match(title)
    if match:
        numbering = match.group(1).strip()
        title_content = match.group(3).lstrip()
        title = numbering + " " + title_content

    title = title.rstrip(".")
    level = (
        title.count(
            ".",
        )
        + 1
        if "." in title
        else 1
    )
    return f"#{'#' * level} {title}".replace("-\n", "").replace(
        "\n",
        " ",
    )


def format_para_title_func(block):
    """Normalize paragraph title, using title_level if available."""
    if not hasattr(block, "title_level"):
        return format_title_func(block)
    level = block.title_level
    title = block.content
    return f"#{'#' * level} {title}".replace("-\n", "").replace(
        "\n",
        " ",
    )


def format_centered_by_html(string, remove_symbol=True):
    if remove_symbol:
        string = string.replace("-\n", "").replace("\n", " ")
    return f'<div style="text-align: center;">{string}</div>' + "\n"


def format_text_plain_func(block):
    return block.content


def format_image_scaled_by_html_func(
    block, original_image_width, show_ocr_content=False
):
    img_tags = []
    if block.image is None:
        return ""
    image_path = block.image["path"]
    image_width = block.bbox[2] - block.bbox[0]
    scale = int(image_width / original_image_width * 100)
    img_tags.append(
        '<img src="{}" alt="Image" width="{}%" />'.format(
            image_path.replace("-\n", "").replace("\n", " "), scale
        ),
    )
    image_info = "\n".join(img_tags)
    if show_ocr_content:
        ocr_content = block.content
        image_info += "\n\n" + ocr_content + "\n\n"
    return image_info


def format_image_plain_func(block, show_ocr_content=False):
    img_tags = []
    if block.image:
        image_path = block.image["path"]
        img_tags.append(
            "![]({})".format(image_path.replace("-\n", "").replace("\n", " "))
        )
        image_info = "\n".join(img_tags)
        if show_ocr_content:
            ocr_content = block.content
            image_info += "\n\n" + ocr_content + "\n\n"
        return image_info
    return ""


def format_chart2markdown_table_func(block):
    """Chart → Markdown table (used by PP-StructureV3 / result_v2)."""
    lines_list = block.content.split("\n")
    column_num = len(lines_list[0].split("|"))
    lines_list.insert(1, "|".join(["---"] * column_num))
    lines_list = [f"|{line}|" for line in lines_list]
    return "\n".join(lines_list)


def format_chart2html_table_func(block):
    """Chart → HTML table (used by PaddleOCR-VL)."""
    lines_list = block.content.split("\n")
    header = lines_list[0].split("|")
    rows = [line.split("|") for line in lines_list[1:]]
    html = "<table border=1 style='margin: auto; width: max-content;'>\n"
    html += (
        "  <thead><tr>"
        + "".join(
            f"<th style='text-align: center;'>{cell.strip()}</th>" for cell in header
        )
        + "</tr></thead>\n"
    )
    html += "  <tbody>\n"
    for row in rows:
        html += (
            "    <tr>"
            + "".join(
                f"<td style='text-align: center;'>{cell.strip()}</td>" for cell in row
            )
            + "</tr>\n"
        )
    html += "  </tbody>\n"
    html += "</table>"
    return html


def simplify_table_func(table_code):
    return "\n" + table_code.replace("<html>", "").replace("</html>", "").replace(
        "<body>", ""
    ).replace("</body>", "")


def format_first_line_func(block, templates, format_func, splitter):
    lines = block.content.split(splitter)
    for idx in range(len(lines)):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.lower() in templates:
            lines[idx] = format_func(line)
        break
    return splitter.join(lines)


def format_table_center_func(block):
    """Add center styling to table HTML (used by PaddleOCR-VL)."""
    table_content = block.content
    table_content = table_content.replace(
        "<table>", "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    )
    table_content = table_content.replace(
        "<th>", "<th style='text-align: center; word-wrap: break-word;'>"
    )
    table_content = table_content.replace(
        "<td>", "<td style='text-align: center; word-wrap: break-word;'>"
    )
    return table_content


def merge_formula_and_number(formula, formula_number):
    """Merge a formula and its formula number for display."""
    formula = formula.replace("$$", "")
    merge_formula = r"{} \tag*{{{}}}".format(formula, formula_number)
    return f"$${merge_formula}$$"


# ---------------------------------------------------------------------------
# build_handle_funcs_dict — unified label→handler mapping
# ---------------------------------------------------------------------------


def build_handle_funcs_dict(
    *,
    text_func,
    image_func,
    chart_func,
    table_func,
    formula_func,
    seal_func,
    use_plain_header_footer_image=False,
):
    """Build a dictionary mapping block labels to their formatting functions.

    Args:
        text_func: Function to format text blocks.
        image_func: Function to format image blocks.
        chart_func: Function to format chart blocks.
        table_func: Function to format table blocks.
        formula_func: Function to format formula blocks.
        seal_func: Function to format seal blocks.
        use_plain_header_footer_image: If True, header_image/footer_image use
            format_image_plain_func instead of image_func (result_v2 behavior).

    Returns:
        dict: A mapping from block label to handler function.
    """
    header_footer_image_func = (
        format_image_plain_func if use_plain_header_footer_image else image_func
    )
    return {
        "paragraph_title": format_para_title_func,
        "abstract_title": format_title_func,
        "reference_title": format_title_func,
        "content_title": format_title_func,
        "doc_title": lambda block: f"# {block.content}".replace("-\n", "").replace(
            "\n", " "
        ),
        "table_title": text_func,
        "figure_title": text_func,
        "chart_title": text_func,
        "vision_footnote": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "text": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "ocr": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "vertical_text": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "reference_content": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "abstract": partial(
            format_first_line_func,
            templates=["摘要", "abstract"],
            format_func=lambda l: f"## {l}\n",
            splitter=" ",
        ),
        "content": lambda block: block.content.replace("-\n", "  \n").replace(
            "\n", "  \n"
        ),
        "image": image_func,
        "chart": chart_func,
        "formula": formula_func,
        "display_formula": formula_func,
        "inline_formula": formula_func,
        "table": table_func,
        "reference": partial(
            format_first_line_func,
            templates=["参考文献", "references"],
            format_func=lambda l: f"## {l}",
            splitter="\n",
        ),
        "algorithm": lambda block: block.content.strip("\n"),
        "seal": seal_func,
        "spotting": lambda block: block.content,
        "number": format_text_plain_func,
        "footnote": format_text_plain_func,
        "header": format_text_plain_func,
        "header_image": header_footer_image_func,
        "footer": format_text_plain_func,
        "footer_image": header_footer_image_func,
        "aside_text": format_text_plain_func,
    }
