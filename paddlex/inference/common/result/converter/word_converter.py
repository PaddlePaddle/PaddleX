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

"""WordConverter — converts structured word_blocks to a docx.Document."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


def _set_paragraph_style(para, config):
    """Apply font/alignment config to a Word paragraph."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    run = para.runs[0] if para.runs else para.add_run()
    font_name = config.get("font", "Times New Roman")
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(config.get("size", 12))
    run.bold = config.get("bold", False)
    para.alignment = config.get("align", WD_ALIGN_PARAGRAPH.LEFT)
    if config.get("indent", False):
        para.paragraph_format.first_line_indent = Inches(0.3)


def _parse_html_table(html: str) -> List[List[str]]:
    """Parse an HTML table into a list of rows (each row is a list of cell texts)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return [
        [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        for tr in soup.find_all("tr")
    ]


def build_word_blocks(
    parsing_res_list: List[Any],
    extra_style_map: Optional[Dict[str, Dict]] = None,
) -> tuple:
    """Build word_blocks and images list from a parsing_res_list.

    Extracts the shared logic for converting DocumentBlock / PaddleOCRVLBlock
    objects into the word_blocks format expected by WordConverter.convert().

    Args:
        parsing_res_list: List of block objects with .label, .content, .image attrs.
        extra_style_map: Optional dict of label->style overrides merged on top of
            BASE_STYLE_MAP via dict.update(). Use for pipeline-specific labels.

    Returns:
        Tuple of (word_blocks, images) where:
            word_blocks: List[Dict] with keys "type", "content", "config".
            images: List[Dict] with keys "path" and "img".
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    BASE_STYLE_MAP = {
        "doc_title": {
            "level": 0,
            "size": 20,
            "bold": True,
            "align": WD_ALIGN_PARAGRAPH.CENTER,
        },
        "header": {
            "size": 16,
            "bold": True,
            "align": WD_ALIGN_PARAGRAPH.CENTER,
        },
        "abstract_title": {
            "level": 1,
            "size": 14,
            "bold": True,
            "align": WD_ALIGN_PARAGRAPH.CENTER,
        },
        "content_title": {
            "level": 1,
            "size": 14,
            "bold": True,
            "align": WD_ALIGN_PARAGRAPH.LEFT,
        },
        "reference_title": {
            "level": 1,
            "size": 14,
            "bold": True,
            "align": WD_ALIGN_PARAGRAPH.LEFT,
        },
        "paragraph_title": {
            "level": 2,
            "size": 14,
            "bold": True,
            "align": WD_ALIGN_PARAGRAPH.LEFT,
        },
        "abstract": {"size": 12, "align": WD_ALIGN_PARAGRAPH.JUSTIFY},
        "text": {
            "size": 12,
            "align": WD_ALIGN_PARAGRAPH.JUSTIFY,
            "indent": True,
        },
        "figure_title": {"size": 10, "align": WD_ALIGN_PARAGRAPH.CENTER},
        "table_title": {"size": 10, "align": WD_ALIGN_PARAGRAPH.CENTER},
        "chart_title": {"size": 10, "align": WD_ALIGN_PARAGRAPH.CENTER},
        "reference": {"size": 12, "align": WD_ALIGN_PARAGRAPH.JUSTIFY},
        "algorithm": {
            "font": "Courier New",
            "size": 11,
            "align": WD_ALIGN_PARAGRAPH.LEFT,
        },
        "formula": {"size": 12, "align": WD_ALIGN_PARAGRAPH.CENTER},
        "vision_footnote": {"size": 9, "align": WD_ALIGN_PARAGRAPH.LEFT},
        "number": {"size": 9, "align": WD_ALIGN_PARAGRAPH.CENTER},
        "footer": {"size": 9, "align": WD_ALIGN_PARAGRAPH.CENTER},
    }

    style_map = {**BASE_STYLE_MAP}
    if extra_style_map:
        style_map.update(extra_style_map)

    default_config = {"size": 12, "align": WD_ALIGN_PARAGRAPH.LEFT, "indent": True}

    word_blocks = []
    images = []

    for block in parsing_res_list:
        label = block.label
        content = getattr(block, "content", "")
        if label in ["image", "chart", "seal"]:
            if block.image is None:
                continue
            content = block.image["path"]
        config = style_map.get(label, default_config)
        word_blocks.append(
            {
                "type": label,
                "content": deepcopy(content),
                "config": config,
            }
        )
        if block.image is not None:
            images.append({"path": block.image["path"], "img": block.image["img"]})

    return word_blocks, images


class WordConverter:
    """Convert structured word_blocks to a :class:`docx.Document`."""

    @staticmethod
    def convert(
        word_blocks: List[Dict],
        *,
        abs_image_paths: Dict[str, str],
        original_image_width: int = 500,
    ):
        """Convert word_blocks to a docx.Document object.

        Args:
            word_blocks: List[Dict] — each dict has keys "type", "content", "config",
                optional "page_index".
            abs_image_paths: Dict[str, str] — {original_path: abs_path} from save_images().
            original_image_width: int — reserved for future scaling (currently unused).

        Returns:
            docx.Document
        """
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches

        doc = Document()
        current_page = None

        for block in word_blocks:
            page_idx = block.get("page_index", 0)
            if current_page is None:
                current_page = page_idx
            elif page_idx != current_page:
                doc.add_section()
                current_page = page_idx

            label = block.get("type")
            content = block.get("content", "").strip()
            config = block.get("config") or {}

            # --- header/footer ---
            if label == "header" and content:
                section = doc.sections[-1]
                section.header.is_linked_to_previous = False
                para = section.header.add_paragraph(content)
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif label == "footer" and content:
                section = doc.sections[-1]
                section.footer.is_linked_to_previous = False
                para = section.footer.add_paragraph(content)
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # --- image/chart/seal ---
            if label in ["chart", "image", "seal"]:
                image_name = block.get("content")
                if not image_name:
                    continue
                abs_image_path = abs_image_paths.get(image_name)
                if not abs_image_path:
                    continue
                para = doc.add_paragraph()
                run = para.add_run()
                run.add_picture(abs_image_path, width=Inches(5))
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER

            # --- table ---
            elif label == "table" and content:
                rows = (
                    _parse_html_table(content)
                    if "<table" in content
                    else [r.split("\t") for r in content.split("\n") if r.strip()]
                )
                if rows:
                    max_cols = max(len(r) for r in rows)
                    table = doc.add_table(rows=0, cols=max_cols)
                    table.style = "Table Grid"
                    for row_cells in rows:
                        row = table.add_row().cells
                        for i in range(max_cols):
                            row[i].text = (
                                row_cells[i].strip() if i < len(row_cells) else ""
                            )

            # --- other text content ---
            elif (
                label
                not in [
                    "header",
                    "footer",
                    "table",
                    "chart",
                    "image",
                    "seal",
                    "vision_footnote",
                ]
                and content
            ):
                para = doc.add_paragraph(content)
                _set_paragraph_style(para, config)

        return doc
