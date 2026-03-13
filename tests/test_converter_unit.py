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

"""Unit tests for the converter module (format_funcs + MarkdownConverter)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from paddlex.inference.common.result.converter.format_funcs import (
    build_handle_funcs_dict,
    format_centered_by_html,
    format_chart2html_table_func,
    format_chart2markdown_table_func,
    format_first_line_func,
    format_image_plain_func,
    format_image_scaled_by_html_func,
    format_para_title_func,
    format_table_center_func,
    format_text_plain_func,
    format_title_func,
    merge_formula_and_number,
    simplify_table_func,
)
from paddlex.inference.common.result.converter.markdown_converter import (
    MarkdownConverter,
)


# ---------------------------------------------------------------------------
# MockBlock
# ---------------------------------------------------------------------------


class MockBlock:
    """Minimal DocumentBlock protocol implementation for testing."""

    def __init__(self, label, bbox, content="", image=None, **kwargs):
        self.label = label
        self.bbox = list(map(int, bbox))
        self.content = content
        self.image = image
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Tests for individual format functions
# ---------------------------------------------------------------------------


def test_format_title_func_with_numbering():
    block = MockBlock("content_title", [0, 0, 100, 20], content="1.2 Introduction")
    result = format_title_func(block)
    assert result.startswith("##"), f"Expected ## prefix, got: {result}"
    assert "Introduction" in result


def test_format_title_func_without_numbering():
    block = MockBlock("doc_title", [0, 0, 100, 20], content="My Document")
    result = format_title_func(block)
    assert result.startswith("##"), f"Expected ## prefix, got: {result}"
    assert "My Document" in result


def test_format_para_title_func_with_title_level():
    block = MockBlock(
        "paragraph_title", [0, 0, 100, 20], content="Section A", title_level=3
    )
    result = format_para_title_func(block)
    assert result.startswith("####"), f"Expected #### for level 3, got: {result}"


def test_format_para_title_func_without_title_level():
    block = MockBlock("paragraph_title", [0, 0, 100, 20], content="1.1 Intro")
    result = format_para_title_func(block)
    # Falls back to format_title_func
    assert result.startswith("##"), f"Expected ## prefix, got: {result}"


def test_format_centered_by_html():
    result = format_centered_by_html("hello world")
    assert '<div style="text-align: center;">hello world</div>' in result


def test_format_centered_by_html_no_remove():
    result = format_centered_by_html("line1-\nline2", remove_symbol=False)
    assert "-\n" in result


def test_format_text_plain_func():
    block = MockBlock("text", [0, 0, 100, 20], content="plain text")
    assert format_text_plain_func(block) == "plain text"


def test_format_image_scaled_by_html_func():
    block = MockBlock("image", [0, 0, 500, 100], image={"path": "img.png", "img": None})
    result = format_image_scaled_by_html_func(block, original_image_width=1000)
    assert 'width="50%"' in result
    assert "img.png" in result


def test_format_image_scaled_by_html_func_none_image():
    block = MockBlock("image", [0, 0, 500, 100], image=None)
    result = format_image_scaled_by_html_func(block, original_image_width=1000)
    assert result == ""


def test_format_image_plain_func():
    block = MockBlock(
        "image", [0, 0, 500, 100], image={"path": "test.png", "img": None}
    )
    result = format_image_plain_func(block)
    assert "![](test.png)" in result


def test_format_image_plain_func_no_image():
    block = MockBlock("image", [0, 0, 500, 100], image=None)
    result = format_image_plain_func(block)
    assert result == ""


def test_format_chart2markdown_table_func():
    block = MockBlock("chart", [0, 0, 100, 100], content="A|B\n1|2\n3|4")
    result = format_chart2markdown_table_func(block)
    assert "|A|B|" in result
    assert "|---|---|" in result


def test_format_chart2html_table_func():
    block = MockBlock("chart", [0, 0, 100, 100], content="A|B\n1|2")
    result = format_chart2html_table_func(block)
    assert "<table" in result
    assert "<th" in result
    assert "<td" in result


def test_simplify_table_func():
    result = simplify_table_func("<html><body><table>content</table></body></html>")
    assert "<html>" not in result
    assert "<body>" not in result
    assert "<table>content</table>" in result


def test_format_table_center_func():
    block = MockBlock(
        "table", [0, 0, 100, 100], content="<table><th>A</th><td>1</td></table>"
    )
    result = format_table_center_func(block)
    assert "border=1" in result
    assert "text-align: center" in result


def test_merge_formula_and_number():
    result = merge_formula_and_number("$$E=mc^2$$", "(1)")
    assert "E=mc^2" in result
    assert r"\tag*{(1)}" in result
    assert result.startswith("$$")
    assert result.endswith("$$")


def test_format_first_line_func():
    block = MockBlock(
        "abstract", [0, 0, 100, 100], content="摘要 This is abstract text"
    )
    result = format_first_line_func(
        block,
        templates=["摘要", "abstract"],
        format_func=lambda l: f"## {l}\n",
        spliter=" ",
    )
    assert "## 摘要" in result


# ---------------------------------------------------------------------------
# Tests for build_handle_funcs_dict
# ---------------------------------------------------------------------------


def test_build_handle_funcs_dict_keys():
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=lambda b: "",
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: "",
        seal_func=lambda b: "",
    )
    expected_labels = {
        "paragraph_title",
        "abstract_title",
        "reference_title",
        "content_title",
        "doc_title",
        "table_title",
        "figure_title",
        "chart_title",
        "vision_footnote",
        "text",
        "ocr",
        "vertical_text",
        "reference_content",
        "abstract",
        "content",
        "image",
        "chart",
        "formula",
        "display_formula",
        "inline_formula",
        "table",
        "reference",
        "algorithm",
        "seal",
        "spotting",
        "number",
        "footnote",
        "header",
        "header_image",
        "footer",
        "footer_image",
        "aside_text",
    }
    assert set(funcs.keys()) == expected_labels


def test_build_handle_funcs_dict_plain_header_footer():
    """When use_plain_header_footer_image=True, header_image/footer_image use format_image_plain_func."""
    custom_image_func = lambda b: "CUSTOM_IMAGE"
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=custom_image_func,
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: "",
        seal_func=lambda b: "",
        use_plain_header_footer_image=True,
    )
    # header_image should NOT be the custom_image_func
    assert funcs["header_image"] is not custom_image_func
    assert funcs["footer_image"] is not custom_image_func
    # image should still be the custom func
    assert funcs["image"] is custom_image_func


def test_build_handle_funcs_dict_default_header_footer():
    """By default, header_image/footer_image use the passed image_func."""
    custom_image_func = lambda b: "CUSTOM_IMAGE"
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=custom_image_func,
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: "",
        seal_func=lambda b: "",
    )
    assert funcs["header_image"] is custom_image_func
    assert funcs["footer_image"] is custom_image_func


# ---------------------------------------------------------------------------
# Tests for MarkdownConverter
# ---------------------------------------------------------------------------


def test_markdown_converter_basic():
    blocks = [
        MockBlock("doc_title", [0, 0, 100, 20], content="Title"),
        MockBlock("text", [0, 30, 100, 60], content="Paragraph one."),
        MockBlock("text", [0, 70, 100, 100], content="Paragraph two."),
    ]
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=lambda b: "",
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: "",
        seal_func=lambda b: "",
    )
    result = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict=funcs,
        page_index=0,
        input_path="test.png",
    )
    assert "Title" in result["markdown_texts"]
    assert "Paragraph one." in result["markdown_texts"]
    assert "Paragraph two." in result["markdown_texts"]
    assert result["page_index"] == 0
    assert result["input_path"] == "test.png"
    assert "page_continuation_flags" not in result


def test_markdown_converter_with_seg_flag():
    blocks = [
        MockBlock("text", [0, 0, 100, 20], content="Part A"),
        MockBlock("text", [0, 30, 100, 60], content="Part B"),
    ]
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=lambda b: "",
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: "",
        seal_func=lambda b: "",
    )
    # get_seg_flag that always says "continuation" (start=False)
    result = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict=funcs,
        use_seg_flag=True,
        get_seg_flag_func=lambda block, prev: (False, True),
    )
    md = result["markdown_texts"]
    # With seg_flag=False for consecutive text, no \n\n between them
    assert "Part APart B" in md or md == "Part APart B"
    assert "page_continuation_flags" in result


def test_markdown_converter_image_collection():
    blocks = [
        MockBlock(
            "image", [0, 0, 100, 100], image={"path": "fig1.png", "img": "DATA1"}
        ),
        MockBlock("text", [0, 110, 100, 130], content="Caption"),
    ]
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=lambda b: f"![img]({b.image['path']})" if b.image else "",
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: "",
        seal_func=lambda b: "",
    )
    result = MarkdownConverter.convert(blocks, handle_funcs_dict=funcs)
    assert result["markdown_images"]["fig1.png"] == "DATA1"


def test_markdown_converter_formula_number_merging():
    blocks = [
        MockBlock("formula", [0, 0, 100, 20], content="$$E=mc^2$$"),
        MockBlock("formula_number", [110, 0, 150, 20], content="(1)"),
        MockBlock("text", [0, 30, 100, 60], content="Explanation"),
    ]
    funcs = build_handle_funcs_dict(
        text_func=lambda b: b.content,
        image_func=lambda b: "",
        chart_func=lambda b: "",
        table_func=lambda b: "",
        formula_func=lambda b: b.content,
        seal_func=lambda b: "",
    )
    result = MarkdownConverter.convert(
        blocks,
        handle_funcs_dict=funcs,
        show_formula_number=True,
    )
    # The formula block content should have been merged with the number
    assert r"\tag*{(1)}" in result["markdown_texts"]


def test_markdown_converter_imgs_in_doc():
    blocks = [MockBlock("text", [0, 0, 100, 20], content="Hello")]
    funcs = {"text": lambda b: b.content}
    extra_imgs = [{"path": "extra.png", "img": "EXTRA_DATA"}]
    result = MarkdownConverter.convert(
        blocks, handle_funcs_dict=funcs, imgs_in_doc=extra_imgs
    )
    assert result["markdown_images"]["extra.png"] == "EXTRA_DATA"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
