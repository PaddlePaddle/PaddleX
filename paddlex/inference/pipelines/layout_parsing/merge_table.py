# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Acknowledgement: The logic for table merging in this function is adapted from MinerU.


def full_to_half(text: str) -> str:
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return "".join(result)


def calculate_table_total_columns(soup):
    """
    calculate total columns including colspan and rowspan, accounting for merged cells
    """
    rows = soup.find_all("tr")
    if not rows:
        return 0
    max_cols = 0
    occupied = {}
    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])
        if row_idx not in occupied:
            occupied[row_idx] = {}
        for cell in cells:
            while col_idx in occupied[row_idx]:
                col_idx += 1
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True
            col_idx += colspan
            max_cols = max(max_cols, col_idx)
    return max_cols


def calculate_row_columns(row):
    """
    Calculate the actual number of columns in a single row
    """
    return sum(int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"]))


def calculate_visual_columns(row):
    """
    Calculate the visual number of columns in a single row, excluding colspan (merged cells count as one)
    """
    return len(row.find_all(["td", "th"]))


def detect_table_headers(soup1, soup2, max_header_rows=5):
    """
    Determine how many identical rows exist at the beginning of two tables
    """
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    # Check only the minimum number of rows
    min_rows = min(len(rows1), len(rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    for i in range(min_rows):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])
        if len(cells1) != len(cells2):
            headers_match = header_rows > 0
            break
        # If column counts match, check if content is identical
        match = True
        for c1, c2 in zip(cells1, cells2):
            text1 = "".join(full_to_half(c1.get_text()).split())
            text2 = "".join(full_to_half(c2.get_text()).split())
            if text1 != text2 or int(c1.get("colspan", 1)) != int(c2.get("colspan", 1)):
                match = False
                break
        # Complete match, increment matched row count. Otherwise, stop matching.
        if match:
            header_rows += 1
        else:
            headers_match = header_rows > 0
            break
    if header_rows == 0:
        headers_match = False
    return header_rows, headers_match


def check_rows_match(soup1, soup2):
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    if not rows1 or not rows2:
        return False
    last_row = rows1[-1]
    header_count, _ = detect_table_headers(soup1, soup2)
    first_data_row = rows2[header_count] if len(rows2) > header_count else None
    if not first_data_row:
        return False
    last_cols = calculate_row_columns(last_row)
    first_cols = calculate_row_columns(first_data_row)
    last_visual = calculate_visual_columns(last_row)
    first_visual = calculate_visual_columns(first_data_row)
    return last_cols == first_cols or last_visual == first_visual


def is_skippable(block, allowed_labels):

    continue_keywords = ["continue", "continued", "cont'd", "续", "cont‘d", "續"]

    if block.label in allowed_labels:
        return True

    b_text = str(getattr(block, "text", "") or "").lower()
    b_fig_title = str(getattr(block, "figure_title", "") or "").lower()
    b_doc_title = str(getattr(block, "doc_title", "") or "").lower()
    b_para_title = str(getattr(block, "paragraph_title", "") or "").lower()

    full_content = f"{b_text} {b_fig_title} {b_doc_title} {b_para_title}"

    if any(kw in full_content for kw in continue_keywords):
        return True

    return False


def can_merge_tables(prev_page, prev_block, curr_page, curr_block):

    from bs4 import BeautifulSoup

    x0, y0, x1, y1 = prev_block.bbox
    prev_width = x1 - x0
    x2, y2, x3, y4 = curr_block.bbox
    curr_width = x3 - x2
    if curr_width == 0 or prev_width == 0:
        return False, None, None
    if abs(curr_width - prev_width) / min(curr_width, prev_width) >= 0.1:
        return False, None, None

    prev_index = prev_page.index(prev_block)
    allowed_follow = all(
        b.label in ["footer", "vision_footnote", "number", "footnote", "footer_image"]
        for b in prev_page[prev_index + 1 :]
    )
    if not allowed_follow:
        return False, None, None

    curr_index = curr_page.index(curr_block)
    curr_allowed_labels = ["header", "header_image", "number"]

    allowed_before = all(
        is_skippable(b, curr_allowed_labels) for b in curr_page[:curr_index]
    )
    if not allowed_before:
        return False, None, None

    html_prev = prev_block.content
    html_curr = curr_block.content
    if not html_prev or not html_curr:
        return False, None, None
    soup_prev = BeautifulSoup(html_prev, "html.parser")
    soup_curr = BeautifulSoup(html_curr, "html.parser")

    total_cols_prev = calculate_table_total_columns(soup_prev)
    total_cols_curr = calculate_table_total_columns(soup_curr)
    tables_match = total_cols_prev == total_cols_curr
    rows_match = check_rows_match(soup_prev, soup_curr)

    return (tables_match or rows_match), soup_prev, soup_curr


def perform_table_merge(soup_prev, soup_curr):
    header_count, _ = detect_table_headers(soup_prev, soup_curr)
    rows_prev = soup_prev.find_all("tr")
    rows_curr = soup_curr.find_all("tr")
    for row in rows_curr[header_count:]:
        row.extract()
        rows_prev[-1].parent.append(row)
    return str(soup_prev)


def merge_tables_across_pages(pages):
    nums = 0
    # get the length of each page
    page_lens = [len(page) for page in pages]
    for i in range(len(pages) - 1, 0, -1):
        page_curr = pages[i]
        page_prev = pages[i - 1]

        for block in page_curr:
            if block.label == "table":
                curr_block = block
                break
        else:
            curr_block = None

        for block in reversed(page_prev):
            if block.label == "table":
                prev_block = block
                break
        else:
            prev_block = None

        # both curr_block and prev_block should not be None
        if curr_block and prev_block:
            can_merge, soup_prev, soup_curr = can_merge_tables(
                page_prev, prev_block, page_curr, curr_block
            )
        else:
            can_merge = False

        if can_merge:
            merged_html = perform_table_merge(soup_prev, soup_curr)
            prev_block.content = merged_html
            curr_block.content = ""
            # one table spilt into more than two pages, the group_id should be the same
            if curr_block.group_id is not None:
                prev_block.group_id = curr_block.group_id
            else:
                new_id = pages[i - 1].index(prev_block) + sum(page_lens[: i - 1])
                prev_block.group_id = new_id
                curr_block.group_id = new_id
    return pages
