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

import json
import os
import re
from bs4 import BeautifulSoup

def escape_latex(s: str) -> str:
    """
    转义 LaTeX 特殊字符（对普通文本使用）
    """
    if not s:
        return ""
    return (s.replace('\\', '\\textbackslash{}')
             .replace('&', '\\&')
             .replace('%', '\\%')
             .replace('$', '\\$')
             .replace('#', '\\#')
             .replace('_', '\\_')
             .replace('{', '\\{')
             .replace('}', '\\}')
             .replace('~', '\\textasciitilde{}')
             .replace('^', '\\textasciicircum{}'))

def get_image_width_from_md(md_path: str, image_name: str, default_ratio: float = 0.8) -> float:
    """
    从 Markdown/HTML 文件中提取 <img> 的 width（仅解析百分比，如 10% -> 0.10）。
    支持 width="10%"、width=10%、style="width:10%;" 等格式。
    返回 0 < ratio <= 1.0，找不到则返回 default_ratio。
    """
    if not md_path or not os.path.exists(md_path):
        return default_ratio

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    flags = re.I | re.DOTALL

    # 优先匹配 width 属性
    pat_width_attr = re.compile(
        rf'<img\s+[^>]*src\s*=\s*["\']?[^"\'>]*{re.escape(image_name)}[^"\'>]*["\']?[^>]*\bwidth\s*=\s*["\']?(\d+)\s*%?["\']?',
        flags
    )
    m = pat_width_attr.search(content)
    if m:
        try:
            val = int(m.group(1))
            if val > 0:
                return min(max(val / 100.0, 0.01), 1.0)
        except:
            pass

    return default_ratio

def escaped_paragraph_text(s: str) -> str:
    """
    处理 text block：
    - 普通文本转义
    - 保留公式原样
    """
    paragraphs = re.split(r'\n\s*\n', s)
    processed_paras = []

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # 临时占位公式
        placeholders = []

        def placeholder_repl(m):
            placeholders.append(m.group(0))
            return f"@@FORMULA{len(placeholders)-1}@@"

        formula_pattern = re.compile(r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\])', re.DOTALL)
        temp_text = formula_pattern.sub(placeholder_repl, p)
        temp_text = escape_latex(temp_text)

        # 替换回公式
        for i, formula in enumerate(placeholders):
            temp_text = temp_text.replace(f"@@FORMULA{i}@@", formula)

        processed_paras.append("\\par " + temp_text)

    return "\n\n".join(processed_paras) + "\n\n"

def generate_image_latex(block, image_base_path, md_base_path) -> str:
    """
    图像/表格处理函数
    """
    bbox = block.get("block_bbox", [0, 0, 0, 0])
    try:
        x1, y1, x2, y2 = map(int, bbox)
    except:
        x1, y1, x2, y2 = 0, 0, 0, 0

    filenames = [
        f"img_in_chart_box_{x1}_{y1}_{x2}_{y2}.jpg",
        f"img_in_image_box_{x1}_{y1}_{x2}_{y2}.jpg",
        block.get("file_name") or block.get("image") or ""
    ]
    image_path = None
    for fname in filenames:
        if fname:
            candidate = os.path.join(image_base_path or "", fname)
            if os.path.exists(candidate):
                image_path = os.path.abspath(candidate)
                break

    if not image_path:
        return f"% [Image not found: {filenames[0]} / {filenames[1]}]\n\n"

    caption_text = escape_latex(block.get("caption", "").strip())

    # 获取宽度
    width_ratio = 0.8
    if md_base_path:
        page = int(block.get("page", 0) or 0)
        md_candidates = [
            os.path.join(md_base_path, f"page_{page}.md"),
            os.path.join(md_base_path, f"document_sample_{page}.md"),
            os.path.join(md_base_path, f"{page}.md"),
        ]
        for mdp in md_candidates:
            if os.path.exists(mdp):
                width_ratio = get_image_width_from_md(mdp, os.path.basename(image_path), default_ratio=width_ratio)
                break
    width_ratio = max(0.01, min(float(width_ratio), 1.0))

    return (
        f"\\begin{{figure}}[h]\n"
        f"\\centering\n"
        f"\\includegraphics[width={width_ratio:.2f}\\linewidth]{{{image_path}}}\n"
        f"\\caption*{{{caption_text}}}\n"
        f"\\end{{figure}}\n\n"
    )

def generate_table_latex(block) -> str:
    content = block.get("block_content", "")
    if "<table" in content:
        soup = BeautifulSoup(content, "html.parser")
        rows = [
            [escape_latex(td.get_text(strip=True)) if not re.search(r'(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])', td.get_text(strip=True)) else td.get_text(strip=True)
             for td in tr.find_all(["td", "th"])]
            for tr in soup.find_all("tr")
        ]
    else:
        rows = [
            [escape_latex(c) if not re.search(r'(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])', c) else c for c in row.split("\t")]
            for row in content.splitlines() if row.strip()
        ]

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    norm_rows = [r + [""] * (col_count - len(r)) for r in rows]
    col_format = " ".join([">{\\raggedright\\arraybackslash}X" for _ in range(col_count)])

    latex = "\\begin{center}\n\\renewcommand{\\arraystretch}{1.5}\n"
    latex += f"\\begin{{tabularx}}{{\\textwidth}}{{{col_format}}}\n\\toprule\n"
    for i, row in enumerate(norm_rows):
        latex += " & ".join(row) + " \\\\\n"
        if i == 0:
            latex += "\\midrule\n"
    latex += "\\bottomrule\n\\end{tabularx}\n\\end{center}\n\n"
    return latex

def block_to_latex(block: dict, image_base_path: str = None, md_base_path: str = None):
    """
    单个 block 转 LaTeX
    """
    label = block.get("block_label", "")
    raw_content = block.get("block_content", "") or ""

    if label == "doc_title":
        content = escape_latex(raw_content.strip())
        return f"\\begin{{center}}\n{{\\Huge {content}}}\\end{{center}}\n\n", None

    if label in ["header", "footer"]:
        return "", None

    if label == "abstract":
        content = raw_content.strip()
        if not content:
            return "", None
        return f"\\begin{{abstract}}\n{escape_latex(content)}\n\\end{{abstract}}\n\n", None

    if label == "paragraph_title":
        content = escape_latex(raw_content.strip())
        return f"\\section*{{{content}}}\n\n", None

    if label == "reference":
        lines = [line.strip() for line in raw_content.split("\n") if line.strip()]
        bibitems = []
        for line in lines:
            content = escape_latex(re.sub(r'^\[\d+\]\s*', '', line))
            key = f"ref{abs(hash(line)) % 100000}"
            bibitems.append(f"\\bibitem{{{key}}} {content}")
        return "\n".join(bibitems) + "\n", "\n".join(bibitems) + "\n"

    if label == "text":
        return escaped_paragraph_text(raw_content), None
    if label == "content":
        lines = [line.rstrip() for line in raw_content.splitlines()]
        latex_lines_content = [escape_latex(line) + " \\\\" for line in lines if line.strip()]
        return "\n".join(latex_lines_content) + "\n\n", None

    if label == "formula":
        return f"\\[\n{raw_content.strip()}\n\\]\n\n", None

    if label == "algorithm":
        return "\\begin{verbatim}\n" + raw_content + "\n\\end{verbatim}\n\n", None

    if label in ["image", "chart", "seal"]:
        return generate_image_latex(block, image_base_path, md_base_path), None

    if label == "table":
        return generate_table_latex(block), None

    if label in ["figure_title", "chart_title", "table_title"]:
        content = escape_latex(raw_content.strip())
        if not content:
            return "", None
        return f"\\begin{{center}}\n{{\\small {content}}}\\end{{center}}\n\n", None

    return f"% 未知标签 {label}: {escape_latex(raw_content)}\n\n", None

def blocks_to_latex(json_path: str, tex_output_path: str, image_base_path: str, md_base_path: str = None):
    if not os.path.exists(json_path):
        print(f"❌ JSON 文件不存在: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        try:
            blocks = json.load(f)
        except Exception as e:
            print("❌ 读取 JSON 失败：", e)
            return

    # 分页
    pages = {}
    for b in blocks:
        p = int(b.get("page", 0) or 0)
        pages.setdefault(p, []).append(b)

    # LaTeX 文档头
    latex_lines = [
        "\\documentclass[12pt]{article}",
        "\\usepackage{xeCJK}",
        "\\usepackage{fontspec}",
        "\\usepackage{graphicx}",
        "\\usepackage{amsmath}",
        "\\usepackage{geometry}",
        "\\usepackage{fancyhdr}",
        "\\usepackage{indentfirst}",
        "\\usepackage{caption}",
        "\\usepackage{tabularx, booktabs}",
        "\\usepackage{amssymb}",
        "\\usepackage{amsfonts}",
        "\\geometry{a4paper, margin=1in}",
        "\\setCJKmainfont{Droid Sans Fallback}",
        "\\setmainfont{DejaVu Serif}",
        "\\setsansfont{Lato}",
        "\\setmonofont{Latin Modern Mono}",
        "\\pagestyle{fancy}",
        "\\setlength{\\parindent}{2em}",
        "\\begin{document}\n"
    ]

    in_bibliography = False
    pending_references = []

    for page_num in sorted(pages.keys()):
        page_blocks = sorted(pages[page_num], key=lambda b: (b.get("block_bbox", [0,0,0,0])[1] if b.get("block_bbox") else 0))
        header_blocks = [b for b in page_blocks if b.get("block_label") == "header"]
        footer_blocks = [b for b in page_blocks if b.get("block_label") == "footer"]
        page_header = " ".join(b.get("block_content", "") for b in header_blocks)
        page_footer = " ".join(b.get("block_content", "") for b in footer_blocks)

        latex_lines.append(f"% ==== page {page_num} header/footer ====")
        latex_lines.append(f"\\fancyhead[L]{{{escape_latex(page_header)}}}")
        latex_lines.append(f"\\fancyfoot[C]{{{escape_latex(page_footer)}}}\n")

        for block in page_blocks:
            lbl = block.get("block_label", "")
            if lbl == "reference_title":
                if not in_bibliography:
                    latex_lines.append("\\begin{thebibliography}{99}")
                    in_bibliography = True
                continue

            tex_fragment, bib_item = block_to_latex(block, 
                                                                      image_base_path=image_base_path,
                                                                      md_base_path=md_base_path)
            if lbl == "reference" and bib_item:
                if in_bibliography:
                    latex_lines.append(bib_item)
                else:
                    pending_references.append(bib_item)
                continue
            if tex_fragment:
                latex_lines.append(tex_fragment)

        latex_lines.append("\\clearpage\n")

    # 写入 pending references
    if pending_references:
        if not in_bibliography:
            latex_lines.append("\\begin{thebibliography}{99}")
        latex_lines.extend(pending_references)
        if not in_bibliography:
            latex_lines.append("\\end{thebibliography}\n")
        elif in_bibliography:
            latex_lines.append("\\end{thebibliography}\n")

    latex_lines.append("\\end{document}")
    latex_text = "\n".join(latex_lines)

    os.makedirs(os.path.dirname(tex_output_path), exist_ok=True)
    with open(tex_output_path, "w", encoding="utf-8") as f:
        f.write(latex_text)

    print(f"✅ LaTeX 文件已保存至: {tex_output_path}")
