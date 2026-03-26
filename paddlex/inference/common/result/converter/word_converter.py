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
    # Force single line spacing to prevent default 1.15x from consuming extra vertical space
    para.paragraph_format.line_spacing = 1.0


def _classify_number_position(bbox, page_width, page_height):
    """Classify a 'number' block's semantic role based on its bbox position.

    Args:
        bbox: [x1, y1, x2, y2] bounding box in pixel coordinates.
        page_width: Page width in pixels.
        page_height: Page height in pixels.

    Returns:
        One of: 'header', 'footer', 'aside_text'.
    """
    if not bbox or page_width <= 0 or page_height <= 0:
        return "footer"

    x1, y1, x2, y2 = bbox
    y_center = (y1 + y2) / 2.0
    x_center = (x1 + x2) / 2.0

    # Top 10% → header region
    if y_center < page_height * 0.10:
        return "header"

    # Bottom 10% → footer region
    if y_center > page_height * 0.90:
        return "footer"

    # Left 15% or right 15% (not in header/footer zone) → aside_text
    if x_center < page_width * 0.15 or x_center > page_width * 0.85:
        return "aside_text"

    # Default fallback
    return "footer"


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
    page_width: int = 0,
    page_height: int = 0,
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
        page_width: Page width in pixels, used to classify 'number' blocks.
            0 means unknown (defaults to footer classification).
        page_height: Page height in pixels, used to classify 'number' blocks.
            0 means unknown (defaults to footer classification).

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
        if label in ["image", "seal"]:
            if block.image is None:
                continue
            content = block.image["path"]
        elif label == "chart":
            if block.image is not None:
                content = block.image["path"]
            elif content:
                # VLM chart recognition: pipe-delimited table text → reuse table rendering
                content = content.replace("|", "\t")
                label = "table"
            else:
                continue
        elif label == "number":
            # Classify 'number' blocks by position to reuse header/footer/aside_text paths
            bbox = (
                list(block.bbox)
                if hasattr(block, "bbox") and block.bbox is not None
                else None
            )
            label = _classify_number_position(
                bbox, page_width=page_width, page_height=page_height
            )
        config = style_map.get(label, default_config)
        word_block = {
            "type": label,
            "content": deepcopy(content),
            "config": config,
        }
        if include_bbox:
            if hasattr(block, "bbox") and block.bbox is not None:
                word_block["bbox"] = list(block.bbox)
            if hasattr(block, "page_index") and block.page_index is not None:
                word_block["page_index"] = block.page_index
        word_blocks.append(word_block)
        if block.image is not None:
            images.append({"path": block.image["path"], "img": block.image["img"]})

    return word_blocks, images


def _write_block(
    doc,
    block,
    abs_image_paths,
    original_image_width=500,
    space_before_emu=None,
    left_indent_emu=None,
    usable_width_emu=None,
    max_height_emu=None,
):
    """Write a single word_block to the given docx Document (or container).

    Handles image/chart/seal, table, and text blocks. Header/footer blocks
    are intentionally NOT handled here — callers must write them to
    section.header / section.footer separately.

    Args:
        doc: docx.Document or a document-like container supporting
            add_paragraph() / add_table().
        block: Dict with keys "type", "content", "config".
        abs_image_paths: Dict mapping original image path → absolute path.
        original_image_width: Width of the original page image in pixels, used to
            calculate proportional image width in the Word document.
        space_before_emu: Optional space before this block in EMU.
        left_indent_emu: Optional left indent in EMU (single-column only).
        usable_width_emu: Optional usable page width in EMU for proportional sizing.
        max_height_emu: Optional maximum rendered height in EMU for image scaling.
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Emu, Inches, Pt

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
        if space_before_emu is not None:
            para.paragraph_format.space_before = Emu(space_before_emu)
            para.paragraph_format.space_after = Emu(0)
        run = para.add_run()
        # Calculate proportional width based on bbox ratio
        USABLE_PAGE_WIDTH = 6.0  # inches fallback
        bbox = block.get("bbox")
        if bbox and original_image_width > 0:
            ratio = (bbox[2] - bbox[0]) / original_image_width
            if usable_width_emu:
                img_width = max(
                    Inches(1.0), min(int(ratio * usable_width_emu), usable_width_emu)
                )
                # Apply max_height_emu constraint (aspect-ratio preserving)
                if max_height_emu and max_height_emu > 0:
                    try:
                        from PIL import Image as _PILImage

                        _img = _PILImage.open(abs_image_path)
                        natural_w, natural_h = _img.size
                        _img.close()
                        if natural_w > 0 and natural_h > 0:
                            rendered_h = int(img_width * natural_h / natural_w)
                            if rendered_h > max_height_emu:
                                img_width = int(max_height_emu * natural_w / natural_h)
                                img_width = max(Inches(0.5), img_width)
                    except Exception:
                        pass
                run.add_picture(abs_image_path, width=img_width)
            else:
                img_width = max(1.0, min(ratio * USABLE_PAGE_WIDTH, USABLE_PAGE_WIDTH))
                run.add_picture(abs_image_path, width=Inches(img_width))
        else:
            run.add_picture(abs_image_path, width=Inches(5.0))
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # --- table ---
    elif label == "table" and content:
        rows = (
            _parse_html_table(content)
            if "<table" in content
            else [r.split("\t") for r in content.split("\n") if r.strip()]
        )
        if rows:
            # Insert spacer paragraph for spacing before table
            if space_before_emu is not None and space_before_emu > 0:
                spacer = doc.add_paragraph()
                spacer.paragraph_format.space_before = Emu(space_before_emu)
                spacer.paragraph_format.space_after = Emu(0)
                spacer.paragraph_format.line_spacing = Pt(1)
                run = spacer.add_run()
                run.font.size = Pt(1)

            max_cols = max(len(r) for r in rows)
            table = doc.add_table(rows=0, cols=max_cols)
            table.style = "Table Grid"

            # Set proportional table width from bbox
            bbox = block.get("bbox")
            if bbox and original_image_width > 0 and usable_width_emu:
                ratio = (bbox[2] - bbox[0]) / original_image_width
                table_width = max(Inches(2), int(ratio * usable_width_emu))
                col_width = table_width // max_cols
                for col in table.columns:
                    col.width = col_width

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
            "aside_text",
            "vision_footnote",
        ]
        and content
    ):
        para = doc.add_paragraph(content)
        _set_paragraph_style(para, config)
        if space_before_emu is not None:
            para.paragraph_format.space_before = Emu(space_before_emu)
            para.paragraph_format.space_after = Emu(0)
        if left_indent_emu is not None:
            para.paragraph_format.left_indent = Emu(left_indent_emu)


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

    Blocks with labels in LAYOUT_EXCLUDE_LABELS are excluded from the column
    detection pass. Among those, 'seal' and 'formula_number' are re-inserted
    into the correct segment after layout is determined.

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
    # Labels excluded from column detection projection
    LAYOUT_EXCLUDE_LABELS = {
        "header",
        "footer",
        "header_image",
        "footer_image",
        "aside_text",
        "seal",
        "number",
        "formula_number",
    }
    # Among excluded labels, these need to be re-inserted into segments
    # (they are body content, just shouldn't interfere with column detection)
    REINSERT_LABELS = {"seal", "formula_number"}

    def _y_center(b):
        bbox = b.get("bbox")
        return (bbox[1] + bbox[3]) / 2.0 if bbox else 0.0

    def _x_center(b):
        bbox = b.get("bbox")
        return (bbox[0] + bbox[2]) / 2.0 if bbox else 0.0

    # Separate excluded blocks; among them, identify which need re-insertion
    excluded = [b for b in blocks if b.get("type", "") in LAYOUT_EXCLUDE_LABELS]
    reinsert_blocks = [b for b in excluded if b.get("type", "") in REINSERT_LABELS]

    # Filter header/footer/aside_text/number blocks — use only body blocks for detection
    body_blocks = [b for b in blocks if b.get("type", "") not in LAYOUT_EXCLUDE_LABELS]
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
            segments.append(
                {"type": "single", "blocks": seg_blocks, "_y": (y_start, y_end)}
            )
            continue

        # Detect columns from narrow blocks via X-axis gaps
        x_gaps = _find_projection_gaps(narrow, axis=0, length=page_width)

        # Filter out edge gaps (page margins), which are not column dividers.
        # A gap that starts at or very near x=0 (left margin) or ends at or
        # very near x=page_width (right margin) is a margin, not a column gap.
        margin_thresh = max(1, int(page_width * 0.08))
        interior_gaps = [
            g
            for g in x_gaps
            if g[0] > margin_thresh and g[1] < page_width - margin_thresh
        ]
        if interior_gaps:
            x_gaps = interior_gaps

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
                segments.append(
                    {"type": "single", "blocks": seg_blocks, "_y": (y_start, y_end)}
                )
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
            segments.append(
                {"type": "single", "blocks": seg_blocks, "_y": (y_start, y_end)}
            )
            continue

        # Prepend full-span blocks as a separate single segment before this strip
        if full_span:
            segments.append(
                {
                    "type": "single",
                    "blocks": sorted(full_span, key=_y_center),
                    "_y": (y_start, y_end),
                }
            )

        if num_cols == 2:
            seg_type = "dual"
        elif num_cols == 3:
            seg_type = "triple"
        else:
            seg_type = "triple"  # capped at max_cols
            col_blocks = col_blocks[:3]

        segments.append(
            {
                "type": seg_type,
                "columns": col_blocks,
                "_y": (y_start, y_end),
                "_dividers": dividers,
                "_x_gaps": x_gaps,
            }
        )

    # ---- Step 2b: Re-insert seal/formula_number blocks into correct segments ----
    for rb in reinsert_blocks:
        rb_y = _y_center(rb)
        rb_x = _x_center(rb)
        # Find the best matching segment (closest y range)
        best_seg = None
        best_dist = float("inf")
        for seg in segments:
            seg_y_start, seg_y_end = seg.get("_y", (0, page_height))
            if seg_y_start <= rb_y <= seg_y_end:
                best_seg = seg
                best_dist = 0
                break
            dist = min(abs(rb_y - seg_y_start), abs(rb_y - seg_y_end))
            if dist < best_dist:
                best_dist = dist
                best_seg = seg

        if best_seg is None:
            # Fallback: append to last segment
            if segments:
                best_seg = segments[-1]
            else:
                segments.append(
                    {"type": "single", "blocks": [rb], "_y": (0, page_height)}
                )
                continue

        if best_seg["type"] == "single":
            # Insert in y-sorted position
            lst = best_seg["blocks"]
            insert_pos = len(lst)
            for i, b in enumerate(lst):
                if _y_center(b) > rb_y:
                    insert_pos = i
                    break
            lst.insert(insert_pos, rb)
        else:
            # Multi-column: assign to column based on x_center
            dividers = best_seg.get("_dividers", [])
            col_idx = 0
            for div in dividers:
                if rb_x > div:
                    col_idx += 1
                else:
                    break
            col_idx = min(col_idx, len(best_seg["columns"]) - 1)
            lst = best_seg["columns"][col_idx]
            insert_pos = len(lst)
            for i, b in enumerate(lst):
                if _y_center(b) > rb_y:
                    insert_pos = i
                    break
            lst.insert(insert_pos, rb)

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


def _build_page_metrics(body_blocks, page_width_px, page_height_px):
    """Compute layout metrics for one page.

    Args:
        body_blocks: List of block dicts (header/footer/aside_text already excluded).
        page_width_px: Original page width in pixels.
        page_height_px: Original page height in pixels.

    Returns:
        Dict with keys:
            scale_x, scale_y: float (px to EMU)
            content_bbox: (x1, y1, x2, y2) in px — body content bounding box
            margins: (left, right, top, bottom) in EMU
            usable_width_emu: int — page usable width after margins
    """
    # A4: 210mm x 297mm = 7560820 x 10693400 EMU
    PAGE_WIDTH_EMU = 7560820
    PAGE_HEIGHT_EMU = 10693400
    MIN_MARGIN_EMU = 274320  # 0.3 inch
    MAX_MARGIN_EMU = 1828800  # 2.0 inch

    scale_x = PAGE_WIDTH_EMU / max(page_width_px, 1)
    scale_y = PAGE_HEIGHT_EMU / max(page_height_px, 1)

    blocks_with_bbox = [b for b in body_blocks if b.get("bbox")]
    if not blocks_with_bbox:
        # Default margins: 1 inch on all sides
        default_margin = 914400
        usable = PAGE_WIDTH_EMU - 2 * default_margin
        usable_h = PAGE_HEIGHT_EMU - 2 * default_margin
        return {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "content_bbox": None,
            "margins": (default_margin, default_margin, default_margin, default_margin),
            "usable_width_emu": usable,
            "usable_height_emu": max(usable_h, 1),
        }

    x1s = [b["bbox"][0] for b in blocks_with_bbox]
    y1s = [b["bbox"][1] for b in blocks_with_bbox]
    x2s = [b["bbox"][2] for b in blocks_with_bbox]
    y2s = [b["bbox"][3] for b in blocks_with_bbox]

    content_x1, content_y1 = min(x1s), min(y1s)
    content_x2, content_y2 = max(x2s), max(y2s)

    left_px = content_x1
    right_px = max(0, page_width_px - content_x2)
    top_px = content_y1
    bottom_px = max(0, page_height_px - content_y2)

    def _clamp(val_emu):
        return max(MIN_MARGIN_EMU, min(MAX_MARGIN_EMU, val_emu))

    left_m = _clamp(int(left_px * scale_x))
    right_m = _clamp(int(right_px * scale_x))
    top_m = _clamp(int(top_px * scale_y))
    bottom_m = _clamp(int(bottom_px * scale_y))

    usable_width_emu = PAGE_WIDTH_EMU - left_m - right_m
    usable_height_emu = PAGE_HEIGHT_EMU - top_m - bottom_m

    return {
        "scale_x": scale_x,
        "scale_y": scale_y,
        "content_bbox": (content_x1, content_y1, content_x2, content_y2),
        "margins": (left_m, right_m, top_m, bottom_m),
        "usable_width_emu": max(usable_width_emu, 1),
        "usable_height_emu": max(usable_height_emu, 1),
    }


def _compute_vertical_spacing(blocks, scale_y):
    """Compute space_before (EMU) for each block based on y-gap from previous block.

    Args:
        blocks: List of block dicts with "bbox" key, sorted by y.
        scale_y: Pixels-to-EMU conversion factor for Y axis.

    Returns:
        List of int|None, same length as blocks. None means use default spacing.
        First block always returns 0.
    """
    MAX_SPACE_EMU = 914400  # 1 inch cap
    QUANTIZE_STEP = 38100  # 3pt quantization to reduce OCR bbox noise

    spacings = []
    prev_y2 = None
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            spacings.append(None)
            prev_y2 = None
            continue
        y1, y2 = bbox[1], bbox[3]
        if prev_y2 is None:
            spacings.append(0)
        else:
            gap_px = max(0, y1 - prev_y2)
            space_emu = int(gap_px * scale_y)
            space_emu = min(space_emu, MAX_SPACE_EMU)
            # Quantize to 3pt steps to reduce OCR noise
            space_emu = round(space_emu / QUANTIZE_STEP) * QUANTIZE_STEP
            spacings.append(space_emu)
        prev_y2 = y2
    return spacings


def _estimate_block_height(block, column_width_emu, abs_image_paths, scale_x, scale_y):
    """Estimate the rendered height of a single block in Word (EMU).

    Args:
        block: Block dict with "type", "content", "config", optional "bbox".
        column_width_emu: Available column width in EMU.
        abs_image_paths: Dict mapping image name to absolute path.
        scale_x: Pixels-to-EMU X factor.
        scale_y: Pixels-to-EMU Y factor.

    Returns:
        int: Estimated height in EMU.
    """
    import math

    label = block.get("type", "")
    bbox = block.get("bbox")
    config = block.get("config") or {}
    content = block.get("content", "")
    if isinstance(content, str):
        content = content.strip()

    LINE_HEIGHT_FACTOR = 1.2  # Word line height ≈ font_size × 1.2

    if label in ("chart", "image", "seal"):
        image_name = block.get("content")
        abs_path = abs_image_paths.get(image_name) if image_name else None
        if abs_path and bbox and column_width_emu > 0:
            try:
                from PIL import Image as _PILImage

                _img = _PILImage.open(abs_path)
                natural_w, natural_h = _img.size
                _img.close()
                # Replicate _write_block width calculation
                original_image_width_px = max(1, int(column_width_emu / scale_x))
                ratio = (bbox[2] - bbox[0]) / max(original_image_width_px, 1)
                from docx.shared import Inches

                img_width = max(
                    Inches(1.0), min(int(ratio * column_width_emu), column_width_emu)
                )
                rendered_h = int(img_width * natural_h / max(natural_w, 1))
                return max(rendered_h, 914400 // 10)  # min 0.1"
            except Exception:
                pass
        # Fallback: bbox-based
        if bbox:
            return int((bbox[3] - bbox[1]) * scale_y)
        return int(Inches(2.0))  # type: ignore[return-value]

    if label == "table":
        if bbox:
            bbox_h_px = bbox[3] - bbox[1]
            bbox_w_px = max(1, bbox[2] - bbox[0])
            original_height_emu = int(bbox_h_px * scale_y)
            bbox_w_emu = int(bbox_w_px * scale_x)
            inflation = (
                bbox_w_emu / column_width_emu
                if column_width_emu > 0 and bbox_w_emu > column_width_emu
                else 1.0
            )
            return int(original_height_emu * inflation * 1.3)
        return 914400  # 1 inch fallback

    # Text blocks
    if bbox:
        bbox_h_px = bbox[3] - bbox[1]
        bbox_w_px = max(1, bbox[2] - bbox[0])
        original_height_emu = int(bbox_h_px * scale_y)
        bbox_w_emu = int(bbox_w_px * scale_x)
        inflation = (
            bbox_w_emu / column_width_emu
            if column_width_emu > 0 and bbox_w_emu > column_width_emu
            else 1.0
        )
        return int(original_height_emu * inflation * LINE_HEIGHT_FACTOR)

    # No bbox — char-count based estimate
    font_size_emu = int(config.get("size", 12) * 12700)
    if column_width_emu > 0 and font_size_emu > 0:
        chars_per_line = max(1, column_width_emu / (font_size_emu * 0.52))
        num_lines = max(1, math.ceil(len(content) / chars_per_line))
    else:
        num_lines = max(1, len(content) // 80 + 1)
    return int(num_lines * font_size_emu * LINE_HEIGHT_FACTOR)


def _estimate_page_content_height(
    segments, page_metrics, abs_image_paths, scale_y, x_gap_cols=None
):
    """Estimate total vertical content height for one page (EMU).

    Args:
        segments: List of segment dicts from _xy_cut_segment().
        page_metrics: Dict from _build_page_metrics().
        abs_image_paths: Dict mapping image name to absolute path.
        scale_y: Pixels-to-EMU Y factor.
        x_gap_cols: Optional list of (col_widths_emu, gap_widths_emu) per segment,
            for accurate multi-column width. If None, use equal-width split.

    Returns:
        int: Estimated total height in EMU.
    """
    scale_x = page_metrics["scale_x"]
    usable_width_emu = page_metrics["usable_width_emu"]
    total = 0

    for seg_idx, segment in enumerate(segments):
        seg_type = segment["type"]

        if seg_type == "single":
            blocks = segment["blocks"]
            spacings = _compute_vertical_spacing(blocks, scale_y)
            seg_height = 0
            for block, sp in zip(blocks, spacings):
                seg_height += _estimate_block_height(
                    block, usable_width_emu, abs_image_paths, scale_x, scale_y
                )
                if sp:
                    seg_height += sp
            total += seg_height
        else:
            # Multi-column: take the tallest column
            columns = segment["columns"]
            num_cols = len(columns)

            # Compute per-column width from _x_gaps if available
            x_gaps = segment.get("_x_gaps", [])
            col_widths_emu = []
            if x_gaps and len(x_gaps) == num_cols - 1:
                # Replicate the same logic as convert_v2()
                page_width_px = max(
                    (b["bbox"][2] for col in columns for b in col if b.get("bbox")),
                    default=1000,
                )
                col_edges = []
                prev_end = 0
                for gap_start, gap_end in x_gaps:
                    col_edges.append((prev_end, gap_start))
                    prev_end = gap_end + 1
                col_edges.append((prev_end, page_width_px))
                col_widths_px = [max(1, e - s) for s, e in col_edges]
                gap_widths_px = [g[1] - g[0] for g in x_gaps]
                total_px = sum(col_widths_px) + sum(gap_widths_px)
                if total_px > 0:
                    px_to_emu = usable_width_emu / total_px
                    col_widths_emu = [int(w * px_to_emu) for w in col_widths_px]

            if not col_widths_emu:
                # Equal-width fallback
                col_w = usable_width_emu // max(num_cols, 1)
                col_widths_emu = [col_w] * num_cols

            col_heights = []
            for col_idx, col_blocks in enumerate(columns):
                col_w = (
                    col_widths_emu[col_idx]
                    if col_idx < len(col_widths_emu)
                    else col_widths_emu[-1]
                )
                spacings = _compute_vertical_spacing(col_blocks, scale_y)
                ch = 0
                for block, sp in zip(col_blocks, spacings):
                    ch += _estimate_block_height(
                        block, col_w, abs_image_paths, scale_x, scale_y
                    )
                    if sp:
                        ch += sp
                col_heights.append(ch)
            total += max(col_heights) if col_heights else 0

    # Section break overhead: each CONTINUOUS break ≈ 1 line (12pt ≈ 152400 EMU)
    section_break_count = max(0, len(segments) - 1)
    total += section_break_count * 152400

    return total


def _compute_horizontal_indent(block, content_x1_px, page_width_px, scale_x):
    """Compute left_indent (EMU) for a single-column block.

    Only applies indent when the block's left edge is significantly offset
    from the content area's left edge (more than 3% of page width).
    Centered blocks (by config) are skipped.

    Args:
        block: Block dict with "bbox" and "config".
        content_x1_px: X coordinate of the content area left edge in pixels.
        page_width_px: Page width in pixels.
        scale_x: Pixels-to-EMU conversion factor for X axis.

    Returns:
        int or None: left_indent in EMU, or None for no indent.
    """
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    config = block.get("config") or {}
    if config.get("align") == WD_ALIGN_PARAGRAPH.CENTER:
        return None

    bbox = block.get("bbox")
    if not bbox:
        return None

    block_x1 = bbox[0]
    offset_px = block_x1 - content_x1_px
    threshold_px = page_width_px * 0.03
    if offset_px < threshold_px:
        return None

    indent_emu = int(offset_px * scale_x)
    indent_emu = min(indent_emu, 2743200)  # 3 inch cap
    return indent_emu if indent_emu > 0 else None


def _set_section_columns(
    section, num_cols=1, space=720, col_widths_twips=None, gap_widths_twips=None
):
    """Set the number of columns in a section via direct XML manipulation.

    python-docx 1.2.0 has no native multi-column API, so we operate on
    the sectPr XML element directly.

    Args:
        section: docx.section.Section object.
        num_cols: Number of columns (1 = single column, 2 = two columns).
        space: Space between columns in twips (default 720 = 0.5 inch).
            Used only when col_widths_twips is None.
        col_widths_twips: Optional list of individual column widths in twips.
            When provided, creates unequal-width columns.
        gap_widths_twips: Optional list of gap widths (length = num_cols - 1).
            Used together with col_widths_twips for column spacing.
    """
    from docx.oxml.ns import qn
    from lxml import etree

    sectPr = section._sectPr
    # Remove any existing w:cols element
    for existing in sectPr.findall(qn("w:cols")):
        sectPr.remove(existing)

    cols_elem = etree.SubElement(sectPr, qn("w:cols"))
    cols_elem.set(qn("w:num"), str(num_cols))

    if col_widths_twips and len(col_widths_twips) == num_cols and num_cols > 1:
        # Unequal column widths
        cols_elem.set(qn("w:equalWidth"), "0")
        cols_elem.set(qn("w:space"), "0")
        for i, col_w in enumerate(col_widths_twips):
            col_el = etree.SubElement(cols_elem, qn("w:col"))
            col_el.set(qn("w:w"), str(int(col_w)))
            if i < num_cols - 1:
                gap = (
                    gap_widths_twips[i]
                    if gap_widths_twips and i < len(gap_widths_twips)
                    else space
                )
                col_el.set(qn("w:space"), str(int(gap)))
    elif num_cols > 1:
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
        # Override pPrDefault: set line spacing to single (240 = 1.0x) and after to 0.
        # python-docx default template has w:line="276" (1.15x) which consumes
        # extra vertical space in column breaks and other unstyled paragraphs.
        from docx.oxml.ns import qn as _qn

        styles_element = doc.styles.element
        ppr_default = styles_element.find(".//" + _qn("w:pPrDefault"))
        if ppr_default is not None:
            spacing = ppr_default.find(".//" + _qn("w:spacing"))
            if spacing is not None:
                spacing.set(_qn("w:line"), "240")
                spacing.set(_qn("w:lineRule"), "auto")
                spacing.set(_qn("w:after"), "0")
        first_page = True

        HEADER_FOOTER_LABELS = {
            "header",
            "footer",
            "header_image",
            "footer_image",
            "aside_text",
        }

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

            # Compute page-level layout metrics (scale, margins, usable width)
            page_metrics = _build_page_metrics(body_blocks, page_width, page_height)
            scale_x = page_metrics["scale_x"]
            scale_y = page_metrics["scale_y"]
            usable_width_emu = page_metrics["usable_width_emu"]
            content_bbox = page_metrics["content_bbox"]
            content_x1 = content_bbox[0] if content_bbox else 0

            # Apply page margins to the current section (first section of this page)
            page_section = doc.sections[-1]
            left_m, right_m, top_m, bottom_m = page_metrics["margins"]
            from docx.shared import Emu as _Emu

            page_section.left_margin = _Emu(left_m)
            page_section.right_margin = _Emu(right_m)
            page_section.top_margin = _Emu(top_m)
            page_section.bottom_margin = _Emu(bottom_m)
            # Set page size to A4 to match _build_page_metrics() which uses A4 dimensions.
            # python-docx default is US Letter (11"), causing ~0.69" overflow for A4 content.
            _PAGE_WIDTH_EMU = 7560820  # A4 width  (8.27")
            _PAGE_HEIGHT_EMU = 10693400  # A4 height (11.69")
            page_section.page_width = _Emu(_PAGE_WIDTH_EMU)
            page_section.page_height = _Emu(_PAGE_HEIGHT_EMU)

            # EMU_PER_TWIP = 635 (1 twip = 20 points = 635 EMU)
            EMU_PER_TWIP = 635
            usable_width_twips = usable_width_emu // EMU_PER_TWIP

            # Vertical budget: estimate total content height and compute compression ratio.
            # This prevents single-page content from overflowing into a second page due to
            # Word's text reflow (font metrics, column-width-induced line wrapping, etc.).
            usable_height_emu = page_metrics["usable_height_emu"]
            estimated_height = _estimate_page_content_height(
                segments, page_metrics, abs_image_paths, scale_y
            )
            SAFETY_MARGIN = 0.95  # keep 5% buffer to avoid edge-case overflow
            if estimated_height > usable_height_emu * SAFETY_MARGIN:
                v_scale = (usable_height_emu * SAFETY_MARGIN) / max(estimated_height, 1)
            else:
                v_scale = 1.0
            # When overflow is severe (>15%), also scale down images
            img_height_scale = (v_scale / 0.85) if v_scale < 0.85 else 1.0

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

                # Compute unequal column widths from _x_gaps if available
                col_widths_twips = None
                gap_widths_twips = None
                if num_cols > 1:
                    x_gaps = segment.get("_x_gaps", [])
                    if x_gaps and len(x_gaps) == num_cols - 1:
                        # Build column x-boundaries from gaps
                        col_edges = []
                        prev_end = 0
                        for gap_start, gap_end in x_gaps:
                            col_edges.append((prev_end, gap_start))
                            prev_end = gap_end + 1
                        col_edges.append((prev_end, page_width))

                        col_widths_px = [e - s for s, e in col_edges]
                        gap_widths_px = [g[1] - g[0] for g in x_gaps]
                        total_px = sum(col_widths_px) + sum(gap_widths_px)
                        if total_px > 0:
                            scale = usable_width_twips / total_px
                            col_widths_twips = [
                                max(360, int(w * scale)) for w in col_widths_px
                            ]
                            gap_widths_twips = [
                                max(144, int(w * scale)) for w in gap_widths_px
                            ]

                if first_segment:
                    section = doc.sections[-1]
                    _set_section_columns(
                        section,
                        num_cols=num_cols,
                        col_widths_twips=col_widths_twips,
                        gap_widths_twips=gap_widths_twips,
                    )
                    first_segment = False
                else:
                    section = doc.add_section(WD_SECTION.CONTINUOUS)
                    # Minimize section break paragraph height (avoids default line height overhead)
                    if doc.paragraphs:
                        brk_para = doc.paragraphs[-1]
                        brk_para.paragraph_format.space_before = _Emu(0)
                        brk_para.paragraph_format.space_after = _Emu(0)
                        from docx.shared import Pt as _Pt

                        if not brk_para.runs:
                            brk_para.add_run()
                        brk_para.runs[0].font.size = _Pt(1)
                        brk_para.paragraph_format.line_spacing = _Pt(1)
                    _set_section_columns(
                        section,
                        num_cols=num_cols,
                        col_widths_twips=col_widths_twips,
                        gap_widths_twips=gap_widths_twips,
                    )

                if seg_type == "single":
                    blocks_list = segment["blocks"]
                    spacings = _compute_vertical_spacing(blocks_list, scale_y)
                    for block, spacing in zip(blocks_list, spacings):
                        indent = _compute_horizontal_indent(
                            block, content_x1, page_width, scale_x
                        )
                        adjusted_sp = (
                            int(spacing * v_scale) if spacing is not None else None
                        )
                        max_h = (
                            int(
                                _estimate_block_height(
                                    block,
                                    usable_width_emu,
                                    abs_image_paths,
                                    scale_x,
                                    scale_y,
                                )
                                * img_height_scale
                            )
                            if img_height_scale < 1.0
                            and block.get("type") in ("chart", "image", "seal")
                            else None
                        )
                        _write_block(
                            doc,
                            block,
                            abs_image_paths,
                            original_image_width,
                            space_before_emu=adjusted_sp,
                            left_indent_emu=indent,
                            usable_width_emu=usable_width_emu,
                            max_height_emu=max_h,
                        )
                else:
                    # multi-column: write columns left-to-right, separated by column breaks
                    columns = segment["columns"]
                    for col_idx, col_blocks in enumerate(columns):
                        # Determine per-column width for max_height estimation
                        col_w_emu = usable_width_emu // max(num_cols, 1)
                        if col_widths_twips and col_idx < len(col_widths_twips):
                            col_w_emu = col_widths_twips[col_idx] * EMU_PER_TWIP
                        spacings = _compute_vertical_spacing(col_blocks, scale_y)
                        for block, spacing in zip(col_blocks, spacings):
                            adjusted_sp = (
                                int(spacing * v_scale) if spacing is not None else None
                            )
                            max_h = (
                                int(
                                    _estimate_block_height(
                                        block,
                                        col_w_emu,
                                        abs_image_paths,
                                        scale_x,
                                        scale_y,
                                    )
                                    * img_height_scale
                                )
                                if img_height_scale < 1.0
                                and block.get("type") in ("chart", "image", "seal")
                                else None
                            )
                            _write_block(
                                doc,
                                block,
                                abs_image_paths,
                                original_image_width,
                                space_before_emu=adjusted_sp,
                                usable_width_emu=usable_width_emu,
                                max_height_emu=max_h,
                            )
                        # Insert column break after each column except the last
                        if col_idx < len(columns) - 1 and any(
                            c for c in columns[col_idx + 1 :]
                        ):
                            from docx.enum.text import WD_BREAK

                            para = doc.add_paragraph()
                            para.paragraph_format.space_before = _Emu(0)
                            para.paragraph_format.space_after = _Emu(0)
                            run = para.add_run()
                            run.add_break(WD_BREAK.COLUMN)

        return doc
