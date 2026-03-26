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

"""LatexConverter — converts structured latex_blocks to a LaTeX document string."""

from __future__ import annotations

import re
from typing import Dict, List


def _escape_latex(s: str) -> str:
    """Escape LaTeX special characters."""
    if not s:
        return ""
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _escaped_paragraph_text(s: str) -> str:
    """Process regular paragraphs while preserving formulas."""
    paragraphs = re.split(r"\n\s*\n", s)
    processed = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        # Hold LaTeX/math formulas by replacing them with placeholders
        placeholders = []

        def _hold(m):
            placeholders.append(m.group(0))
            return f"@@FORMULA{len(placeholders)-1}@@"

        temp = re.sub(r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\])", _hold, p, flags=re.DOTALL)
        temp = _escape_latex(temp)
        for i, f in enumerate(placeholders):
            temp = temp.replace(f"@@FORMULA{i}@@", f)
        processed.append("\\par " + temp)
    return "\n\n".join(processed) + "\n\n"


def _generate_image_latex(block: Dict, abs_image_paths: Dict[str, str]) -> str:
    image_name = block.get("content")
    if not image_name:
        return "% [Image not found]\n\n"
    abs_image_path = abs_image_paths.get(image_name)
    if not abs_image_path:
        return f"% [Image path not found for {image_name}]\n\n"
    return (
        f"\\begin{{figure}}[h]\n"
        f"\\centering\n"
        f"\\includegraphics[width=0.8\\linewidth]{{{abs_image_path}}}\n"
        f"\\end{{figure}}\n\n"
    )


def _generate_table_latex(block: Dict) -> str:
    from bs4 import BeautifulSoup

    content = block.get("content", "")
    if "<table" in content:
        soup = BeautifulSoup(content, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            row = []
            for td in tr.find_all(["td", "th"]):
                cell = td.get_text(strip=True)
                row.append(
                    cell
                    if re.search(r"(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])", cell)
                    else _escape_latex(cell)
                )
            rows.append(row)
    else:
        rows = [
            [
                (
                    _escape_latex(c)
                    if not re.search(r"(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])", c)
                    else c
                )
                for c in row.split("\t")
            ]
            for row in content.splitlines()
            if row.strip()
        ]

    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    norm_rows = [r + [""] * (col_count - len(r)) for r in rows]
    col_format = " ".join(
        [">{\\raggedright\\arraybackslash}X" for _ in range(col_count)]
    )

    latex = "\\begin{center}\n\\renewcommand{\\arraystretch}{1.5}\n"
    latex += f"\\begin{{tabularx}}{{\\textwidth}}{{{col_format}}}\n\\toprule\n"
    for i, row in enumerate(norm_rows):
        latex += " & ".join(row) + " \\\\\n"
        if i == 0:
            latex += "\\midrule\n"
    latex += "\\bottomrule\n\\end{tabularx}\n\\end{center}\n\n"
    return latex


def _block_to_latex(block: Dict, abs_image_paths: Dict[str, str]) -> str:
    label = block.get("type", "")
    content = block.get("content", "") or ""
    if label == "doc_title":
        return f"\\begin{{center}}\n{{\\Huge {_escape_latex(content.strip())}}}\\end{{center}}\n\n"
    if label in ["header", "footer"]:
        return ""
    if label == "abstract":
        return f"\\begin{{abstract}}\n{_escape_latex(content.strip())}\n\\end{{abstract}}\n\n"
    if label == "paragraph_title":
        return f"\\section*{{{_escape_latex(content.strip())}}}\n\n"
    if label == "text":
        return _escaped_paragraph_text(content)
    if label == "content":
        lines = [line.rstrip() for line in content.splitlines()]
        return (
            "\n".join([_escape_latex(line) + " \\\\" for line in lines if line.strip()])
            + "\n\n"
        )
    if label == "formula":
        return f"\\[\n{content.strip()}\n\\]\n\n"
    if label == "algorithm":
        return "\\begin{verbatim}\n" + content + "\n\\end{verbatim}\n\n"
    if label in ["image", "chart", "seal"]:
        return _generate_image_latex(block, abs_image_paths)
    if label == "table":
        return _generate_table_latex(block)
    if label in ["figure_title", "table_title", "chart_title"]:
        return f"\\begin{{center}}\n{{\\small {_escape_latex(content.strip())}}}\\end{{center}}\n\n"
    if label == "reference":
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        bibitems = []
        for line in lines:
            escaped = _escape_latex(re.sub(r"^\[\d+\]\s*", "", line))
            key = f"ref{abs(hash(line)) % 100000}"
            bibitems.append(f"\\bibitem{{{key}}} {escaped}")
        return "\n".join(bibitems) + "\n"
    return f"% [Unknown block: {label}] {_escape_latex(content)}\n\n"


class LatexConverter:
    """Convert structured latex_blocks to a complete LaTeX document string."""

    @staticmethod
    def convert(
        latex_blocks: List[Dict],
        *,
        abs_image_paths: Dict[str, str],
    ) -> str:
        """Convert latex_blocks to a complete LaTeX document string.

        Args:
            latex_blocks: List[Dict] — each dict has keys "type", "content",
                optional "page_index", "block_bbox".
            abs_image_paths: Dict[str, str] — from save_images().

        Returns:
            str: Complete LaTeX source with preamble and \\begin{document}...\\end{document}.
        """
        pages: Dict[int, List[Dict]] = {}
        for b in latex_blocks:
            p = int(b.get("page_index", 0) or 0)
            pages.setdefault(p, []).append(b)

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
            "\\begin{document}\n",
        ]

        in_bib = False
        for page_num in sorted(pages.keys()):
            page_blocks = sorted(
                pages[page_num], key=lambda b: b.get("block_bbox", [0, 0, 0, 0])[1]
            )
            header_blocks = [b for b in page_blocks if b.get("type") == "header"]
            footer_blocks = [b for b in page_blocks if b.get("type") == "footer"]
            page_header = " ".join(b.get("content", "") for b in header_blocks)
            page_footer = " ".join(b.get("content", "") for b in footer_blocks)

            latex_lines.append(f"% ==== page {page_num} header/footer ====")
            latex_lines.append(f"\\fancyhead[L]{{{_escape_latex(page_header)}}}")
            latex_lines.append(f"\\fancyfoot[C]{{{_escape_latex(page_footer)}}}\n")

            for block in page_blocks:
                if block.get("type", "") == "reference_title" and not in_bib:
                    latex_lines.append("\\begin{thebibliography}{99}")
                    in_bib = True
                    continue
                latex_lines.append(_block_to_latex(block, abs_image_paths))

            latex_lines.append("\\clearpage\n")

        if in_bib:
            latex_lines.append("\\end{thebibliography}\n")

        latex_lines.append("\\end{document}")
        return "\n".join(latex_lines)
