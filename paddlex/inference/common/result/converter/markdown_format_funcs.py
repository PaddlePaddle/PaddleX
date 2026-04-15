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


def format_title(block):
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
    return _collapse_soft_newlines(f"#{'#' * level} {title}")


def format_para_title(block):
    """Normalize paragraph title, using title_level if available."""
    if getattr(block, "title_level", None) is None:
        return format_title(block)
    level = block.title_level
    title = block.content
    return _collapse_soft_newlines(f"#{'#' * level} {title}")


def format_centered_by_html(content, collapse_newlines=True):
    """Wrap *content* in an HTML center-aligned div.

    Args:
        content: Pre-formatted string (e.g., an ``<img>`` tag or plain text).
            Unlike the other format helpers this function takes a string, not
            a block object.
        collapse_newlines: If *True* (default), collapse soft-hyphen line
            breaks (``"-\\n"`` → ``""``) and hard line breaks (``"\\n"`` →
            ``" "``) before wrapping.  Set to *False* to preserve the string
            as-is (useful for multi-line HTML content).

    Returns:
        str: HTML ``<div style="text-align: center;">…</div>`` followed by a
        newline.
    """
    if collapse_newlines:
        content = _collapse_soft_newlines(content)
    return f'<div style="text-align: center;">{content}</div>' + "\n"


def format_text_plain(block):
    """Return the block's raw text content without any transformation."""
    return block.content


def format_image_scaled_by_html(block, original_image_width, show_ocr_content=False):
    """Render an image block as a width-scaled HTML ``<img>`` tag.

    Unlike the standard ``(block) -> str`` formatters, this function requires
    ``original_image_width`` — a page-level context value — and therefore
    **cannot be used directly as a handler**.  Callers must curry it via a
    ``lambda`` or :func:`functools.partial`::

        format_image_func = lambda block: format_centered_by_html(
            format_image_scaled_by_html(block, original_image_width=width)
        )

    Args:
        block: Document block with ``image``, ``bbox``, and optionally
            ``content`` attributes.
        original_image_width: Full pixel width of the source page image, used
            to compute the block's relative width as a percentage.
        show_ocr_content: If *True*, append ``block.content`` (OCR text) below
            the image tag.

    Returns:
        str: HTML ``<img>`` tag (optionally followed by OCR text), or ``""``
        when ``block.image`` is *None*.
    """
    img_tags = []
    if block.image is None:
        return ""
    image_path = block.image["path"]
    image_width = block.bbox[2] - block.bbox[0]
    scale = int(image_width / original_image_width * 100)
    img_tags.append(
        '<img src="{}" alt="Image" width="{}%" />'.format(
            _collapse_soft_newlines(image_path), scale
        ),
    )
    image_info = "\n".join(img_tags)
    if show_ocr_content:
        ocr_content = block.content
        image_info += "\n\n" + ocr_content + "\n\n"
    return image_info


def format_image_plain(block, show_ocr_content=False):
    """Render an image block as a Markdown image reference ``![](path)``.

    Args:
        block: Document block with an ``image`` dict (``{"path": str, "img":
            PIL.Image}``) or *None*.
        show_ocr_content: If *True*, append ``block.content`` (OCR text) below
            the Markdown image tag.

    Returns:
        str: Markdown image reference, or ``""`` when ``block.image`` is *None*.
    """
    img_tags = []
    if block.image:
        image_path = block.image["path"]
        img_tags.append("![]({})".format(_collapse_soft_newlines(image_path)))
        image_info = "\n".join(img_tags)
        if show_ocr_content:
            ocr_content = block.content
            image_info += "\n\n" + ocr_content + "\n\n"
        return image_info
    return ""


def format_chart2markdown_table(block):
    """Chart → Markdown table (used by PP-StructureV3 / result_v2)."""
    lines_list = block.content.split("\n")
    column_num = len(lines_list[0].split("|"))
    lines_list.insert(1, "|".join(["---"] * column_num))
    lines_list = [f"|{line}|" for line in lines_list]
    return "\n".join(lines_list)


def format_chart2html_table(block):
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


def simplify_table(table_code):
    """Strip ``<html>`` and ``<body>`` wrapper tags from a table HTML string.

    Note: Unlike other format helpers, this function accepts a raw HTML string
    (not a block object) and is typically called inside a lambda::

        format_table_func = lambda block: simplify_table("\\n" + block.content)

    Args:
        table_code: Raw HTML string, typically containing ``<table>`` wrapped
            in ``<html><body>…</body></html>``.

    Returns:
        str: HTML with outer ``<html>`` and ``<body>`` tags removed, preceded
        by a newline.
    """
    return "\n" + table_code.replace("<html>", "").replace("</html>", "").replace(
        "<body>", ""
    ).replace("</body>", "")


def format_first_line(block, templates, format_func, splitter):
    """Format the first non-empty line of a block if it matches a template.

    Intended for use with :func:`functools.partial` to create fixed handlers
    for labels such as ``"abstract"`` or ``"reference"``::

        partial(format_first_line,
                templates=["摘要", "abstract"],
                format_func=lambda l: f"## {l}\\n",
                splitter=" ")

    Args:
        block: Document block whose ``content`` is split by *splitter*.
        templates: List of lowercase strings to match against the first
            non-empty line (case-insensitive).
        format_func: Called with the matching line; its return value replaces
            that line in the output.
        splitter: String delimiter used to split and re-join ``block.content``.

    Returns:
        str: Content with the first matching line reformatted, or the original
        content if no line matches.
    """
    lines = block.content.split(splitter)
    for idx in range(len(lines)):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.lower() in templates:
            lines[idx] = format_func(line)
        break
    return splitter.join(lines)


def format_table_center(block):
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


def _collapse_soft_newlines(s: str) -> str:
    """Collapse soft-hyphen line breaks and newlines into spaces."""
    return s.replace("-\n", "").replace("\n", " ")


def _format_normalize_newlines(block):
    """Normalize double newlines to single, then single to double for markdown spacing."""
    return block.content.replace("\n\n", "\n").replace("\n", "\n\n")


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
            format_image_plain instead of image_func (result_v2 behavior).

    Returns:
        dict: A mapping from block label to handler function.
    """
    header_footer_image_func = (
        format_image_plain if use_plain_header_footer_image else image_func
    )
    return {
        "paragraph_title": format_para_title,
        "abstract_title": format_title,
        "reference_title": format_title,
        "content_title": format_title,
        "doc_title": lambda block: _collapse_soft_newlines(f"# {block.content}"),
        "table_title": text_func,
        "figure_title": text_func,
        "chart_title": text_func,
        "vision_footnote": _format_normalize_newlines,
        "text": _format_normalize_newlines,
        "ocr": _format_normalize_newlines,
        "vertical_text": _format_normalize_newlines,
        "reference_content": _format_normalize_newlines,
        "abstract": partial(
            format_first_line,
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
            format_first_line,
            templates=["参考文献", "references"],
            format_func=lambda l: f"## {l}",
            splitter="\n",
        ),
        "algorithm": lambda block: block.content.strip("\n"),
        "seal": seal_func,
        "spotting": format_text_plain,
        "number": format_text_plain,
        "footnote": format_text_plain,
        "header": format_text_plain,
        "header_image": header_footer_image_func,
        "footer": format_text_plain,
        "footer_image": header_footer_image_func,
        "aside_text": format_text_plain,
    }
