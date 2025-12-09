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
    "CHINESE_NUM": re.compile(r"^\s*(?:第)?([一二三四五六七八九十]+)", flags=re.I),
}


# Extract numbering type and its semantic level
def get_symbol_and_level(content: str):
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
    1: [
        "ABSTRACT",
        "SUMMARY",
        "RESUME",
        "绪论",
        "引言",
        "CONTENTS",
        "REFERENCES",
        "参考文献",
        "APPENDIX",
        "附录",
        "ACKNOWLEDGMENTS",
    ]
}


# Cluster heading heights to infer level based on font size
def cluster_global_heights(entries, k_clusters=4):

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


# Assign a global ordering to different numbering styles
def compute_global_symbol_seq(entries, title_symbol_level):

    seq = {}
    counter = 1

    for e in entries:
        symbol, level = title_symbol_level[e["content"]]

        if level > 0 and symbol not in seq:
            seq[symbol] = counter
            counter += 1

    return seq


# Compute final level for each heading
def compute_levels_for_entries(entries):

    # get title's symbol and level
    title_symbol_level = {}
    for e in entries:
        symbol, level = get_symbol_and_level(e["content"])
        e["symbol"], e["level"] = symbol, level
        title_symbol_level[e["content"]] = (symbol, level)

    cluster_map = cluster_global_heights(entries)
    global_seq = compute_global_symbol_seq(entries, title_symbol_level)

    # Used to align multi-level numeric lists (e.g., "1", "1.1", "1.2")
    first_num_level = 0

    contents = []
    levels = []

    for e in entries:

        content_u = str(e["content"]).upper()

        if e.get("level") == 0:
            continue

        symbol, level = title_symbol_level[e["content"]]

        # if matches the semantics in SYMBOL_PATTERNS,bucket the semantic level
        if level > 0:
            bucket = "Semantic"
        # Check special keywords (ABSTRACT, REFERENCES, etc.)
        elif any(w in content_u for kw in SPECIAL_KEYWORDS.values() for w in kw):
            for level, keywords in SPECIAL_KEYWORDS.items():
                if any(w in content_u for w in keywords):
                    RelativeOrder_level = level
                    break
            bucket = "RelativeOrder"
        else:
            bucket = "Cluster"

        Cluster_level = cluster_map[e["height"]]

        if bucket == "Semantic":
            Semantic_level = level

            if symbol == "NUM_LIST":
                if first_num_level != 0:
                    RelativeOrder_level = global_seq.get(symbol) + (
                        level - first_num_level
                    )
                else:
                    first_num_level = level
                    RelativeOrder_level = global_seq.get(symbol)
            else:
                RelativeOrder_level = global_seq.get(symbol)

            # Voting among three signals
            votes = [Semantic_level, RelativeOrder_level, Cluster_level]
            most_common = Counter(votes).most_common(1)

            if most_common[0][1] > 1:
                final_level = most_common[0][0]
            else:
                final_level = RelativeOrder_level

        elif bucket == "RelativeOrder":
            final_level = RelativeOrder_level

        else:
            final_level = Cluster_level

        e["level"] = int(final_level)

        contents.append(e["content"])
        levels.append(e["level"])

    return entries


# Write computed levels back to the parsing results
def assign_levels_to_parsing_res(parsing_res_list):

    entries = []

    for block in parsing_res_list:

        if block.label not in ("paragraph_title", "doc_title"):
            continue

        content = getattr(block, "content", "")
        bbox = getattr(block, "bbox")
        height = bbox[3] - bbox[1]

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

    if len(entries) == 0:
        return parsing_res_list

    entries = compute_levels_for_entries(entries)

    for e in entries:
        block = e["origin_block"]
        setattr(block, "title_level", e["level"])

    return parsing_res_list
