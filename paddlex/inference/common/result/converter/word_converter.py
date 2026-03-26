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
    include_bbox: bool = False,
) -> tuple:
    """Build word_blocks and images list from a parsing_res_list.

    Extracts the shared logic for converting DocumentBlock / PaddleOCRVLBlock
    objects into the word_blocks format expected by WordConverter.convert().

    Args:
        parsing_res_list: List of block objects with .label, .content, .image attrs.
        extra_style_map: Optional dict of label->style overrides merged on top of
            BASE_STYLE_MAP via dict.update(). Use for pipeline-specific labels.
        include_bbox: If True, include "bbox" field in each word_block dict.
            Defaults to False for backwards compatibility.

    Returns:
        Tuple of (word_blocks, images) where:
            word_blocks: List[Dict] with keys "type", "content", "config",
                and optionally "bbox".
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
        word_block = {
            "type": label,
            "content": deepcopy(content),
            "config": config,
        }
        if include_bbox and hasattr(block, "bbox") and block.bbox is not None:
            word_block["bbox"] = list(block.bbox)
        word_blocks.append(word_block)
        if block.image is not None:
            images.append({"path": block.image["path"], "img": block.image["img"]})

    return word_blocks, images


def _write_block(doc, block, abs_image_paths, original_image_width=500):
    """Write a single word_block to the given docx Document (or container).

    Handles image/chart/seal, table, and text blocks. Header/footer blocks
    are intentionally NOT handled here — callers must write them to
    section.header / section.footer separately.

    Args:
        doc: docx.Document or a document-like container supporting
            add_paragraph() / add_table().
        block: Dict with keys "type", "content", "config".
        abs_image_paths: Dict mapping original image path → absolute path.
        original_image_width: Reserved for future scaling (currently unused).
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches

    label = block.get("type")
    content = block.get("content", "")
    if isinstance(content, str):
        content = content.strip()
    config = block.get("config") or {}

    # --- image/chart/seal ---
    if label in ["chart", "image", "seal"]:
        image_name = block.get("content")
        if not image_name:
            return
        abs_image_path = abs_image_paths.get(image_name)
        if not abs_image_path:
            return
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
                    row[i].text = row_cells[i].strip() if i < len(row_cells) else ""

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


def _is_full_span(block, page_width, threshold=0.6):
    """Determine whether a block spans the full page width (single-column).

    Args:
        block: Dict with optional "bbox" key [x1, y1, x2, y2].
        page_width: Total width of the page in pixels.
        threshold: Width ratio above which a block is considered full-span.

    Returns:
        True if the block should be treated as a full-width (single-column) element.
    """
    bbox = block.get("bbox")
    if not bbox or page_width <= 0:
        return True  # no bbox info → treat as full span to be safe

    x1, y1, x2, y2 = bbox
    block_width = x2 - x1
    ratio = block_width / page_width

    label = block.get("type", "")
    # Title/header/footer elements are considered full-span at a lower threshold
    full_span_labels = {
        "doc_title",
        "header",
        "footer",
        "header_image",
        "footer_image",
    }
    if label in full_span_labels:
        return ratio > 0.45

    return ratio > threshold


def _find_projection_gaps(blocks, axis, length, min_gap_ratio=0.02):
    """Project blocks onto the given axis and find unoccupied gaps.

    Args:
        blocks: List of block dicts, each with a "bbox" key [x1, y1, x2, y2].
        axis: 0 = X axis (find vertical gaps / column dividers),
              1 = Y axis (find horizontal gaps / row dividers).
        length: Total length of the axis (page_width or page_height).
        min_gap_ratio: Minimum gap width as a fraction of total length.

    Returns:
        List of (gap_start, gap_end) tuples for each valid unoccupied interval.
    """
    if length <= 0 or not blocks:
        return []

    min_gap = max(1, int(min_gap_ratio * length))
    occupied = [False] * length

    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        if axis == 0:
            start, end = int(x1), int(x2)
        else:
            start, end = int(y1), int(y2)
        start = max(0, start)
        end = min(length - 1, end)
        for i in range(start, end + 1):
            occupied[i] = True

    gaps = []
    gap_start = None
    for i in range(length):
        if not occupied[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_width = i - gap_start
                if gap_width >= min_gap:
                    gaps.append((gap_start, i - 1))
                gap_start = None
    if gap_start is not None:
        gap_width = length - gap_start
        if gap_width >= min_gap:
            gaps.append((gap_start, length - 1))

    return gaps


def _xy_cut_segment(blocks, page_width, page_height, max_cols=3):
    """Segment page blocks using XY-Cut projection.

    First splits the block set into horizontal strips via Y-axis projection
    gaps, then within each strip detects column count via X-axis projection
    gaps. Adjacent strips of the same type are merged.

    Args:
        blocks: List of block dicts with "bbox" key (header/footer already
            excluded by the caller).
        page_width: Width of the page in pixels.
        page_height: Height of the page in pixels.
        max_cols: Maximum number of columns to detect (default 3).

    Returns:
        List of segment dicts:
            {"type": "single", "blocks": [...]}
            {"type": "dual",   "columns": [[left_blocks], [right_blocks]]}
            {"type": "triple", "columns": [[col1], [col2], [col3]]}
    """
    HEADER_FOOTER_LABELS = {"header", "footer", "header_image", "footer_image"}

    def _y_center(b):
        bbox = b.get("bbox")
        return (bbox[1] + bbox[3]) / 2.0 if bbox else 0.0

    # Filter header/footer blocks
    body_blocks = [b for b in blocks if b.get("type", "") not in HEADER_FOOTER_LABELS]
    if not body_blocks:
        return []

    # ---- Step 1: Y-axis split into horizontal strips ----
    y_gaps = _find_projection_gaps(body_blocks, axis=1, length=page_height)

    # Build list of (y_start, y_end) strip boundaries
    strip_boundaries = []
    prev_y = 0
    for gap_start, gap_end in y_gaps:
        strip_boundaries.append((prev_y, gap_start - 1))
        prev_y = gap_end + 1
    strip_boundaries.append((prev_y, page_height - 1))

    # ---- Step 2: For each horizontal strip, detect columns via X-axis ----
    segments = []
    for y_start, y_end in strip_boundaries:
        # Assign blocks whose y-center falls within this strip
        strip_blocks = [
            b
            for b in body_blocks
            if b.get("bbox") and y_start <= (b["bbox"][1] + b["bbox"][3]) / 2 <= y_end
        ]
        if not strip_blocks:
            # Fall back: any block overlapping this strip's y range
            strip_blocks = [
                b
                for b in body_blocks
                if b.get("bbox")
                and not (b["bbox"][3] < y_start or b["bbox"][1] > y_end)
            ]
        if not strip_blocks:
            continue

        # Separate full-span and narrow blocks in this strip
        full_span = [b for b in strip_blocks if _is_full_span(b, page_width)]
        narrow = [b for b in strip_blocks if not _is_full_span(b, page_width)]

        if not narrow:
            # Only full-span blocks → single segment
            seg_blocks = sorted(strip_blocks, key=_y_center)
            segments.append({"type": "single", "blocks": seg_blocks})
            continue

        # Detect columns from narrow blocks via X-axis gaps
        x_gaps = _find_projection_gaps(narrow, axis=0, length=page_width)

        if not x_gaps or len(x_gaps) + 1 > max_cols:
            # Use the widest gaps as dividers when too many gaps
            if x_gaps and len(x_gaps) + 1 > max_cols:
                x_gaps = sorted(x_gaps, key=lambda g: g[1] - g[0], reverse=True)[
                    : max_cols - 1
                ]
                x_gaps = sorted(x_gaps)
            else:
                # No valid gaps → single column
                seg_blocks = sorted(strip_blocks, key=_y_center)
                segments.append({"type": "single", "blocks": seg_blocks})
                continue

        num_cols = len(x_gaps) + 1
        # Build column x-boundaries: divider is the center of each gap
        dividers = [(g[0] + g[1]) // 2 for g in x_gaps]

        # Assign narrow blocks to columns
        col_blocks = [[] for _ in range(num_cols)]
        for b in narrow:
            bbox = b.get("bbox")
            if not bbox:
                col_blocks[0].append(b)
                continue
            cx = (bbox[0] + bbox[2]) / 2.0
            col_idx = 0
            for div in dividers:
                if cx > div:
                    col_idx += 1
                else:
                    break
            col_blocks[col_idx].append(b)

        # Sort each column by y
        col_blocks = [sorted(col, key=_y_center) for col in col_blocks]

        # If only one column received blocks, treat as single
        non_empty_cols = [col for col in col_blocks if col]
        if len(non_empty_cols) <= 1:
            seg_blocks = sorted(strip_blocks, key=_y_center)
            segments.append({"type": "single", "blocks": seg_blocks})
            continue

        # Prepend full-span blocks as a separate single segment before this strip
        if full_span:
            segments.append(
                {"type": "single", "blocks": sorted(full_span, key=_y_center)}
            )

        if num_cols == 2:
            seg_type = "dual"
        elif num_cols == 3:
            seg_type = "triple"
        else:
            seg_type = "triple"  # capped at max_cols
            col_blocks = col_blocks[:3]

        segments.append({"type": seg_type, "columns": col_blocks})

    # ---- Step 3: Merge adjacent segments of the same type ----
    merged = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        if prev["type"] == seg["type"]:
            if seg["type"] == "single":
                prev["blocks"] = prev["blocks"] + seg["blocks"]
            elif seg["type"] in ("dual", "triple"):
                n_cols = len(seg["columns"])
                if len(prev.get("columns", [])) == n_cols:
                    for i in range(n_cols):
                        prev["columns"][i] = prev["columns"][i] + seg["columns"][i]
                else:
                    merged.append(seg)
            else:
                merged.append(seg)
        else:
            merged.append(seg)

    return merged


def _set_section_columns(section, num_cols=1, space=720):
    """Set the number of columns in a section via direct XML manipulation.

    python-docx 1.2.0 has no native multi-column API, so we operate on
    the sectPr XML element directly.

    Args:
        section: docx.section.Section object.
        num_cols: Number of columns (1 = single column, 2 = two columns).
        space: Space between columns in twips (default 720 = 0.5 inch).
    """
    from docx.oxml.ns import qn
    from lxml import etree

    sectPr = section._sectPr
    # Remove any existing w:cols element
    for existing in sectPr.findall(qn("w:cols")):
        sectPr.remove(existing)

    cols_elem = etree.SubElement(sectPr, qn("w:cols"))
    cols_elem.set(qn("w:num"), str(num_cols))
    if num_cols > 1:
        cols_elem.set(qn("w:space"), str(space))
        cols_elem.set(qn("w:equalWidth"), "1")


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

            _write_block(doc, block, abs_image_paths, original_image_width)

        return doc

    @staticmethod
    def convert_v2(
        word_blocks: List[Dict],
        *,
        abs_image_paths: Dict[str, str],
        original_image_width: int = 500,
        original_image_height: int = 700,
    ):
        """Convert word_blocks to a docx.Document with column layout restoration.

        Detects single/dual/triple-column layout using XY-Cut projection and
        creates appropriate Word sections with multi-column formatting. Falls
        back to single-column if bbox data is missing.

        Args:
            word_blocks: List[Dict] — each dict has keys "type", "content",
                "config", optional "page_index", optional "bbox".
            abs_image_paths: Dict[str, str] — {original_path: abs_path}.
            original_image_width: int — used as page_width for column detection.
            original_image_height: int — used as page_height for row detection.

        Returns:
            docx.Document
        """
        from docx import Document
        from docx.enum.section import WD_SECTION
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        # Check if any block has bbox; if not, fall back to convert()
        has_bbox = any(b.get("bbox") is not None for b in word_blocks)
        if not has_bbox:
            return WordConverter.convert(
                word_blocks,
                abs_image_paths=abs_image_paths,
                original_image_width=original_image_width,
            )

        # Group blocks by page_index
        pages: Dict[int, List[Dict]] = {}
        for block in word_blocks:
            page_idx = block.get("page_index", 0)
            pages.setdefault(page_idx, []).append(block)

        doc = Document()
        first_page = True

        HEADER_FOOTER_LABELS = {"header", "footer", "header_image", "footer_image"}

        for page_idx in sorted(pages.keys()):
            page_blocks = pages[page_idx]
            page_width = original_image_width if original_image_width > 0 else 1000
            page_height = original_image_height if original_image_height > 0 else 700

            # Add page break between pages (except the first)
            if not first_page:
                new_section = doc.add_section(WD_SECTION.NEW_PAGE)
                _set_section_columns(new_section, num_cols=1)
            first_page = False

            # Write header/footer for this page into current section
            for block in page_blocks:
                label = block.get("type", "")
                content = block.get("content", "")
                if isinstance(content, str):
                    content = content.strip()
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

            # Segment the page using XY-Cut projection
            body_blocks = [
                b for b in page_blocks if b.get("type", "") not in HEADER_FOOTER_LABELS
            ]
            segments = _xy_cut_segment(body_blocks, page_width, page_height)

            if not segments:
                continue

            first_segment = True
            for segment in segments:
                seg_type = segment["type"]

                # Determine column count for Word section
                if seg_type == "dual":
                    num_cols = 2
                elif seg_type == "triple":
                    num_cols = 3
                else:
                    num_cols = 1

                if first_segment:
                    section = doc.sections[-1]
                    _set_section_columns(section, num_cols=num_cols)
                    first_segment = False
                else:
                    section = doc.add_section(WD_SECTION.CONTINUOUS)
                    _set_section_columns(section, num_cols=num_cols)

                if seg_type == "single":
                    for block in segment["blocks"]:
                        _write_block(doc, block, abs_image_paths, original_image_width)
                else:
                    # multi-column: write columns left-to-right, separated by column breaks
                    columns = segment["columns"]
                    for col_idx, col_blocks in enumerate(columns):
                        for block in col_blocks:
                            _write_block(
                                doc, block, abs_image_paths, original_image_width
                            )
                        # Insert column break after each column except the last
                        if col_idx < len(columns) - 1 and any(
                            c for c in columns[col_idx + 1 :]
                        ):
                            from docx.enum.text import WD_BREAK

                            para = doc.add_paragraph()
                            run = para.add_run()
                            run.add_break(WD_BREAK.COLUMN)

        return doc
