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

import re
from collections import Counter

import numpy as np

# Regular expressions for detecting heading numbering styles
SYMBOL_PATTERNS = {
    # Matches Roman numerals: I, II, V, X, i., iv), V.
    "ROMAN": re.compile(r"^\s*([IVX]+)(?:[\.．\)\s]|$)", flags=re.I),
    # Matches a single letter: A., B), c., D
    "LETTER": re.compile(r"^\s*([A-Z])(?:[\.．\)\s])", flags=re.I),
    # Matches multi-level numeric numbering: 1, 1.1, 1.2.3, 2.
    "NUM_LIST": re.compile(r"^\s*(\d+(?:\.\d+)*)(?![）)])(?:[\.]?\s*|(?=[A-Z]))"),
    # Matches numeric numbering enclosed in parentheses: (1), (1.1), （2）, （2.3）, 1)
    "NUM_LIST_WITH_BRACKET": re.compile(r"^\s*(?:[\(（])?(\d+(?:\.\d+)*)[\)）]"),
    # Matches Chinese numerals: 一 , 二 , 第一 , 十三
    "CHINESE_NUM": re.compile(
        r"^\s*"
        r"(?:第|[（\(])?"
        r"([一二三四五六七八九十]{1,2})"
        r"(?:"
        r"[章节篇卷部条题讲课回）\)]"
        r"|"
        r"(?![a-zA-Z\u4e00-\u9fa5])"
        r")",
        flags=re.I,
    ),
}


def get_symbol_and_level(content: str):
    """
    Extract numbering type and its semantic level
    """
    txt = str(content).strip()

    if SYMBOL_PATTERNS["NUM_LIST_WITH_BRACKET"].match(txt):
        return "NUM_LIST_BRACKET", 4

    if SYMBOL_PATTERNS["ROMAN"].match(txt):
        return "ROMAN", 1

    if SYMBOL_PATTERNS["CHINESE_NUM"].match(txt):
        return "CHINESE_NUM", 1

    if SYMBOL_PATTERNS["LETTER"].match(txt):
        return "LETTER", 2

    if SYMBOL_PATTERNS["NUM_LIST"].match(txt):
        content = SYMBOL_PATTERNS["NUM_LIST"].match(txt).group(1)
        level = content.count(".") + 1
        return "NUM_LIST", level

    return None, -1


# Special keywords that should be treated as level-1 headings
SPECIAL_KEYWORDS = {
    "ABSTRACT": 1,
    "SUMMARY": 1,
    "RESUME": 1,
    "绪论": 1,
    "引言": 1,
    "CONTENTS": 1,
    "REFERENCES": 1,
    "REFERENCE": 1,
    "参考文献": 1,
    "APPENDIX": 1,
    "APPENDICES": 1,
    "附录": 1,
    "ACKNOWLEDGMENTS": 1,
    "INTRODUCTION": 1,
    "BACKGROUNDANDRELATEDWORK": 1,
    "BACKGROUND": 1,
    "RELATEDWORK": 1,
    "THEORETICALMODELS": 1,
    "DATA": 1,
    "METHOD": 1,
    "METHODS": 1,
    "METHODOLOGY": 1,
    "TOPICANALYSIS": 1,
    "RESULT": 1,
    "RESULTS": 1,
    "DISCUSSION": 1,
    "CONCLUSIONS": 1,
    "CONCLUSION": 1,
    "LIMITATIONS": 1,
    "研究背景": 1,
    "相关工作": 1,
    "研究方法": 1,
    "实验结果": 1,
    "讨论": 1,
    "结论": 1,
    "致谢": 1,
    "目录": 1,
}


def get_title_height(block, layout_det_res):
    """
    Calculate the average height of the dominant text lines within a layout block.
    """

    import math

    import cv2

    if block.label == "doc_title":
        return 0

    page_image = layout_det_res[block.page_index]["input_img"]

    # Round down for top-left
    x1 = int(block.bbox[0])
    y1 = int(block.bbox[1])
    # Round up for bottom-right to ensure full coverage
    x2 = int(math.ceil(block.bbox[2]))
    y2 = int(math.ceil(block.bbox[3]))
    # Boundary clamping: Ensure coordinates do not exceed image dimensions
    h, w = page_image.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    title_image = page_image[y1:y2, x1:x2]
    # Convert to grayscale
    title_image = cv2.cvtColor(title_image, cv2.COLOR_RGB2GRAY)
    # Binarization using Otsu's method
    ret, binary = cv2.threshold(
        title_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Determine orientation based on aspect ratio
    h, w = binary.shape[:2]
    aspect_ratio = w / h

    projection = None

    if aspect_ratio >= 1.0:
        # orizontal text: Project to Y-axis
        projection = np.sum(binary, axis=1)
    else:
        # Vertical text: Project to X-axis
        projection = np.sum(binary, axis=0)

    heights_list = []
    current_height = 0
    in_block = False

    # Signals below this are considered background/gap
    threshold = np.max(projection) * 0.05

    for val in projection:
        if val > threshold:
            # Entering or inside a text line
            in_block = True
            current_height += 1
        else:
            if in_block:
                if current_height > 2:
                    heights_list.append(current_height)
                current_height = 0
            in_block = False

    # Edge Case: If the loop ends while still inside a text block
    if in_block and current_height > 2:
        heights_list.append(current_height)

    # Filter for dominant lines: keep lines that are > 80% of the max height
    max_height = max(heights_list)
    threshold = max_height * 0.8
    big_lines = [h for h in heights_list if h > threshold]
    avg_height = sum(big_lines) / len(big_lines)

    return int(round(avg_height))


def cluster_global_heights(entries, k_clusters=4):
    """
    Cluster heading heights to infer level based on font size
    """

    from sklearn.cluster import KMeans

    heights = [e["height"] for e in entries]
    uniq = sorted(set(heights))

    if len(uniq) == 0:
        return {}

    k = min(k_clusters, len(uniq))

    X = np.array(heights).reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)

    centers = km.cluster_centers_.reshape(-1)

    # Sort centers descending: larger font → higher level
    order = np.argsort(-centers)
    old2new = {int(old): new_idx + 1 for new_idx, old in enumerate(order)}

    mapping = {}
    for h in uniq:
        dists = [abs(h - c) for c in centers]
        old = int(np.argmin(dists))
        mapping[h] = old2new[old]

    return mapping


def compute_global_symbol_seq(entries, title_symbol_level):
    """
    Assign a global ordering to different numbering styles
    """

    seq = {}
    counter = 1

    for idx, e in enumerate(entries):
        symbol, level = title_symbol_level[idx]

        if level > 0 and symbol not in seq:
            seq[symbol] = counter
            counter += 1

    return seq


def compute_levels_for_entries(entries):
    """
    Compute final level for each heading
    """

    # get title's symbol and level
    title_symbol_level = {}
    for idx, e in enumerate(entries):
        symbol, level = get_symbol_and_level(e["content"])
        e["symbol"], e["level"] = symbol, level
        title_symbol_level[idx] = (symbol, level)

    cluster_map = cluster_global_heights(entries)
    global_seq = compute_global_symbol_seq(entries, title_symbol_level)

    # Used to align multi-level numeric lists (e.g., "1", "1.1", "1.2")
    first_num_level = 0

    contents = []
    levels = []

    for idx, e in enumerate(entries):

        if e.get("level") == 0:
            continue

        symbol, level = title_symbol_level[idx]

        # if matches the semantics in SYMBOL_PATTERNS,bucket the semantic level
        if level > 0:
            bucket = "semantic"
        # Check special keywords (ABSTRACT, REFERENCES, etc.)
        elif (
            str(e["content"]).upper().strip().rstrip("：: ").replace(" ", "")
            in SPECIAL_KEYWORDS
        ):
            bucket = "special_word"
        else:
            bucket = "cluster"

        cluster_level = cluster_map[e["height"]]

        if bucket == "semantic":
            semantic_level = level

            if symbol == "NUM_LIST":
                if first_num_level != 0:
                    relative_order_level = global_seq.get(symbol) + (
                        level - first_num_level
                    )
                else:
                    first_num_level = level
                    relative_order_level = global_seq.get(symbol)
            else:
                relative_order_level = global_seq.get(symbol)

            # Voting among three signals
            votes = [semantic_level, relative_order_level, cluster_level]
            most_common = Counter(votes).most_common(1)

            if most_common[0][1] > 1:
                final_level = most_common[0][0]
            else:
                final_level = relative_order_level

        elif bucket == "special_word":
            final_level = SPECIAL_KEYWORDS[
                str(e["content"]).upper().strip().rstrip("：: ").replace(" ", "")
            ]

        else:
            final_level = cluster_level

        e["level"] = int(final_level)

        contents.append(e["content"])
        levels.append(e["level"])

    return entries


def assign_levels_to_parsing_res(blocks_by_page, layout_det_res):
    """
    Write computed levels back to the parsing results
    """

    parsing_res_list = []

    for page_index, one_page_blocks in enumerate(blocks_by_page):
        for block in one_page_blocks:
            setattr(block, "page_index", page_index)
            parsing_res_list.append(block)

    entries = []

    for block in parsing_res_list:

        if block.label == "paragraph_title":
            content = block.content
            height = get_title_height(block, layout_det_res)

            if height is None:
                continue

            # Document title has fixed level 0
            init_level = 0 if block.label == "doc_title" else None

            entries.append(
                {
                    "origin_block": block,
                    "content": content,
                    "height": height,
                    "level": init_level,
                }
            )

    entries = compute_levels_for_entries(entries)

    for e in entries:
        if e["origin_block"].label == "doc_title":
            setattr(block, "title_level", 0)
        block = e["origin_block"]
        setattr(block, "title_level", e["level"])

    return blocks_by_page
