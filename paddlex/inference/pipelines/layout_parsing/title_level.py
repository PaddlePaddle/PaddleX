import re
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans

SYMBOL_PATTERNS = {
    "ROMAN": re.compile(r"^\s*([IVX]+)(?:[\.．\)\s]|$)", flags=re.I),
    "LETTER": re.compile(r"^\s*([A-Z])(?:[\.．\)\s])", flags=re.I),
    "NUM_LIST": re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[\)）]?\s*|(?=[A-Z]))"),
    "NUM_LIST_PATTERN2": re.compile(r"^\s*[\(（](\d+(?:\.\d+)*)[\)）]"),
    "CHINESE_NUM": re.compile(r"^\s*(?:第)?([一二三四五六七八九十]+)", flags=re.I),
}


def get_symbol_and_depth_and_token(content: str):
    txt = str(content).strip()

    m = SYMBOL_PATTERNS["NUM_LIST_PATTERN2"].match(txt)
    if m:
        return "NUM_LIST_BRACKET", 4

    m = SYMBOL_PATTERNS["ROMAN"].match(txt)
    if m:
        return "ROMAN", 1

    m = SYMBOL_PATTERNS["CHINESE_NUM"].match(txt)
    if m:
        return "CHINESE_NUM", 1

    m = SYMBOL_PATTERNS["LETTER"].match(txt)
    if m:
        return "LETTER", 2

    m = SYMBOL_PATTERNS["NUM_LIST"].match(txt)
    if m:
        token = m.group(1)
        depth = max(1, token.count(".") + 1)
        return "NUM_LIST", depth

    return "NONE", 0


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


def cluster_global_heights(entries, k_clusters=4):

    heights = [e["height"] for e in entries]
    uniq = sorted(set(heights))

    # 如果不同高度小于4，则按照高度排序
    if len(uniq) == 0:
        return {}

    k = min(k_clusters, len(uniq))

    X = np.array(heights).reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(X)

    centers = km.cluster_centers_.reshape(-1)

    # 簇中心从大到小排序（大字号就层级高）
    order = np.argsort(-centers)
    old2new = {int(old): new_idx + 1 for new_idx, old in enumerate(order)}

    mapping = {}
    for h in uniq:
        dists = [abs(h - c) for c in centers]
        old = int(np.argmin(dists))
        mapping[h] = old2new[old]

    return mapping


def compute_global_symbol_seq(entries):

    seq = {}
    counter = 1

    for e in entries:
        stype, D = get_symbol_and_depth_and_token(e["content"])

        if D > 0 and stype not in seq:
            seq[stype] = counter
            counter += 1

    return seq


# ---------------------------------------------------------
# 5. 核心：根据标题 entries 计算最终层级
# ---------------------------------------------------------
def compute_levels_for_entries(entries):

    phys_map = cluster_global_heights(entries)
    global_seq = compute_global_symbol_seq(entries)

    first_num_depth = 0

    contents = []
    levels = []

    for e in entries:

        content_u = str(e["content"]).upper()

        if e.get("level") == 0:
            continue

        stype, D = get_symbol_and_depth_and_token(e["content"])

        # --------------------------
        # 桶分类
        # --------------------------
        if D > 0:
            bucket = "A"
        else:
            bucket = "C"

            for lvl, kws in SPECIAL_KEYWORDS.items():
                if any(w in content_u for w in kws):
                    bucket = "B"
                    B_level = lvl
                    break

        # 物理层
        L_phys = phys_map.get(e["height"], 1)

        # -------------------------
        # A 桶（三票机制）
        # -------------------------
        if bucket == "A":
            L_exp = D

            if stype == "NUM_LIST":
                if first_num_depth != 0:
                    L_seq = global_seq.get(stype) + (D - first_num_depth)
                else:
                    first_num_depth = D
                    L_seq = global_seq.get(stype)
            else:
                L_seq = global_seq.get(stype)

            votes = [L_exp, L_seq, L_phys]
            most_common = Counter(votes).most_common(1)

            if most_common[0][1] > 1:
                L_final = most_common[0][0]
            else:
                L_final = L_seq

        # -------------------------
        # B 桶
        # -------------------------
        elif bucket == "B":
            L_final = B_level

        # -------------------------
        # C 桶：纯物理层
        # -------------------------
        else:
            L_final = L_phys

        e["level"] = int(L_final)

        contents.append(e["content"])
        levels.append(e["level"])

    # print(contents)
    # print(levels)

    return entries


# ---------------------------------------------------------
# 6. 针对 parsing_res_list 的完整处理流程
# ---------------------------------------------------------
def assign_levels_to_parsing_res(parsing_res_list):
    """
    parsing_res_list 是一个 LayoutBlock 对象列表
    只处理 label == "paragraph_title" 和 "doc_title"
    """

    entries = []

    for blk in parsing_res_list:

        if blk.label not in ("paragraph_title", "doc_title"):
            continue

        content = getattr(blk, "content", "")
        bbox = getattr(blk, "bbox")
        height = bbox[3] - bbox[1]

        if height is None:
            continue

        init_level = 0 if blk.label == "doc_title" else None

        entries.append(
            {
                "origin_block": blk,
                "content": content,
                "height": height,
                "level": init_level,
            }
        )

    if len(entries) == 0:
        return parsing_res_list

    # -------------------------------------------
    # ② 计算层级
    # -------------------------------------------
    entries = compute_levels_for_entries(entries)

    # -------------------------------------------
    # ③ 写回 LayoutBlock 对象
    # -------------------------------------------
    for e in entries:
        blk = e["origin_block"]
        setattr(blk, "title_level", e["level"])

    return parsing_res_list
