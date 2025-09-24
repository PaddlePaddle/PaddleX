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

import os,re

def escape_latex_outside_formula(s: str) -> str:
    """
    转义 LaTeX 特殊字符，但保留公式原样
    """
    if not s:
        return ""
    
    placeholders = []
    def repl(m):
        placeholders.append(m.group(0))
        return f"@@FORMULA{len(placeholders)-1}@@"
    
    # 提取公式
    formula_pat = re.compile(r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\]|\\\(.*?\\\))', re.DOTALL)
    tmp = formula_pat.sub(repl, s)

    # 转义
    tmp = (tmp.replace('\\', '\\textbackslash{}')
              .replace('&', '\\&')
              .replace('%', '\\%')
              .replace('$', '\\$')
              .replace('#', '\\#')
              .replace('_', '\\_')
              .replace('{', '\\{')
              .replace('}', '\\}')
              .replace('~', '\\textasciitilde{}')
              .replace('^', '\\textasciicircum{}'))

    # 恢复公式
    for i, f in enumerate(placeholders):
        tmp = tmp.replace(f"@@FORMULA{i}@@", f)
    return tmp

def get_image_width_from_md_line(line, default_ratio=0.8):
    """
    解析图片 width 属性
    """
    m = re.search(r'width\s*=\s*["\']?(\d+)%?["\']?', line)
    if m:
        val = int(m.group(1))
        return max(0.01, min(val/100.0, 1.0))
    m2 = re.search(r'width\s*:\s*(\d+)%', line)
    if m2:
        val = int(m2.group(1))
        return max(0.01, min(val/100.0, 1.0))
    return default_ratio

def process_table_html(content) -> str:
    
    from bs4 import BeautifulSoup
    
    """
    表格处理
    """
    if "<table" in content:
        soup = BeautifulSoup(content, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            row = []
            for td in tr.find_all(["td", "th"]):
                text = td.get_text(strip=True)
                row.append(escape_latex_outside_formula(text))
            rows.append(row)
    else:
        rows = [
            [escape_latex_outside_formula(c) for c in row.split("\t")]
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

def process_paragraph(s: str) -> str:
    """
    处理文本段落，保留公式
    """
    paragraphs = re.split(r'\n\s*\n', s)
    processed_paras = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        processed_paras.append("\\par " + escape_latex_outside_formula(p))
    return "\n\n".join(processed_paras) + "\n\n"

def process_md_line(line: str) -> str:
    
    from bs4 import BeautifulSoup
    """
    单行处理
    """
    line = line.strip()
    if not line:
        return ""

    # 标题
    if line.startswith("##### "):
        return f"\\paragraph*{{{escape_latex_outside_formula(line[6:].strip())}}}\n\n"
    if line.startswith("#### "):
        return f"\\subsubsection*{{{escape_latex_outside_formula(line[5:].strip())}}}\n\n"
    if line.startswith("### "):
        return f"\\subsection*{{{escape_latex_outside_formula(line[4:].strip())}}}\n\n"
    if line.startswith("## "):
        return f"\\section*{{{escape_latex_outside_formula(line[3:].strip())}}}\n\n"
    if line.startswith("# "):
        return f"\\section*{{{escape_latex_outside_formula(line[2:].strip())}}}\n\n"

    # 居中 div
    if "<div" in line and "text-align: center" in line:
        soup = BeautifulSoup(line, "html.parser")
        div = soup.find("div")
        if div:
            # 图片
            if div.img:
                img = div.img
                src = img.get("src")
                src = f"/root/wjb/PaddleX-develop/mypaddle/upgit/output/{src}"
                width_ratio = get_image_width_from_md_line(str(img))
                return (f"\\begin{{figure}}[h]\n\\centering\n"
                        f"\\includegraphics[width={width_ratio:.2f}\\linewidth]{{{src}}}\n"
                        f"\\end{{figure}}\n\n")
            # 表格
            if div.table:
                return process_table_html(str(div))
            # 文本
            text = div.get_text(strip=True)
            if text:
                return f"\\begin{{center}}{escape_latex_outside_formula(text)}\\end{{center}}\n\n"

    # 表格
    if "<table" in line:
        return process_table_html(line)

    # 普通段落
    return process_paragraph(line)

def md_to_latex(md_text: str, output_path: str):
    
    from bs4 import BeautifulSoup
    
    pages = [p.strip() for p in re.split(r'<sep>1</sep>', md_text) if p.strip()]
    if not pages:
        print("❌ 没有有效内容")
        return

    # 最后一页作为页眉页脚信息
    meta_info = pages[-1] if "<table" in pages[-1] else ""
    content_pages = pages[:-1] if meta_info else pages
    if len(pages) == 1:
        content_pages = pages
        meta_info = None

    # 页眉页脚提取
    headers, footers, footnotes, page_numbers = [], [], [], []
    if meta_info:
        soup = BeautifulSoup(meta_info, "html.parser")
        table = soup.find("table")
        if table:
            trs = table.find_all("tr")
            for tr in trs[1:]:
                tds = tr.find_all("td")
                headers.append(tds[1].get_text(strip=True))
                footers.append(tds[2].get_text(strip=True))
                footnotes.append(tds[3].get_text(strip=True))
                page_numbers.append(tds[4].get_text(strip=True))

    # LaTeX 文件头
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

    # 逐页处理
    for idx, page in enumerate(content_pages):
        # 页眉页脚
        if headers and idx < len(headers):
            latex_lines.append(f"\\fancyhead[C]{{{escape_latex_outside_formula(headers[idx])}}}")
        if footers and idx < len(footers):
            latex_lines.append(f"\\fancyfoot[C]{{{escape_latex_outside_formula(footers[idx])}}}")

        # 内容处理
        for line in page.splitlines():
            latex_lines.append(process_md_line(line))

        # footnote + page number
        if footnotes and idx < len(footnotes) and footnotes[idx]:
            latex_lines.append(f"\\noindent{{\\small [footnote] {escape_latex_outside_formula(footnotes[idx])}}}\n")
        if page_numbers and idx < len(page_numbers) and page_numbers[idx]:
            latex_lines.append(f"\\begin{{center}}{escape_latex_outside_formula(page_numbers[idx])}\\end{{center}}\n")

        if idx != len(content_pages) - 1:
            latex_lines.append("\\clearpage\n")

    latex_lines.append("\\end{document}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))

    print(f"✅ LaTeX 文件已生成: {output_path}")
