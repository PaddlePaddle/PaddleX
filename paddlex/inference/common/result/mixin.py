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

import copy
import json
import mimetypes
import os
import re
from abc import abstractmethod
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from ....utils import logging
from ...utils.io import (
    CSVWriter,
    HtmlWriter,
    ImageWriter,
    JsonWriter,
    MarkdownWriter,
    TextWriter,
    VideoWriter,
    XlsxWriter,
)


class WordMixin:
    """
    Mixin class for adding Word (.docx) export capabilities.
    """

    def __init__(self, *args: list, **kwargs: dict):
        self._save_funcs.append(self.save_to_word)

    @abstractmethod
    def _to_word(self) -> Dict[str, Any]:
        """
        Convert the result to a Word-compatible format.

        Returns:
            Dict[str, Any]: A dictionary containing Word-compatible blocks and image data.
        """
        raise NotImplementedError

    @property
    def word(self) -> Dict[str, Any]:
        """
        Convert the result to a Word-compatible format.

        Returns:
            Dict[str, Any]: A dictionary containing Word-compatible blocks and image data.
        """
        return self._to_word()

    def save_to_word(self, save_path, *args, **kwargs) -> None:
        """
        Save the Word session dict to a .docx file (each page as separate file).
        """
        from bs4 import BeautifulSoup
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.shared import Inches, Pt

        def save_images(image_list, base_save_path):
            """
            Save images to disk and return a mapping from original path to saved absolute path.
            """
            abs_image_paths: Dict[str, str] = {}
            image_dir = base_save_path / "imgs"
            image_dir.mkdir(parents=True, exist_ok=True)

            for item in image_list:
                img_path = item.get("path")
                img_obj = item.get("img")
                if not img_path or not img_obj:
                    continue

                img_name = Path(img_path).name
                save_path = image_dir / img_name
                img_obj.save(save_path)
                abs_image_paths[img_path] = str(save_path.resolve())
            return abs_image_paths

        def blocks_to_word(word_blocks, original_image_width, abs_image_paths):
            """
            Convert word blocks to Word document format.
            """

            def set_paragraph_style(para, config):
                run = para.runs[0] if para.runs else para.add_run()
                font_name = config.get("font", "Times New Roman")
                run.font.name = font_name
                run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
                run.font.size = Pt(config.get("size", 12))
                run.bold = config.get("bold", False)
                para.alignment = config.get("align", WD_ALIGN_PARAGRAPH.LEFT)
                if config.get("indent", False):
                    para.paragraph_format.first_line_indent = Inches(0.3)

            def parse_html_table(html):
                soup = BeautifulSoup(html, "html.parser")
                return [
                    [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
                    for tr in soup.find_all("tr")
                ]

            doc = Document()
            current_page = None

            for block in word_blocks:
                page_idx = block.get("page_index", 0)
                if current_page is None:
                    current_page = page_idx
                elif page_idx != current_page:
                    # new page -> add a section
                    doc.add_section()
                    current_page = page_idx

                label = block.get("type")
                content = block.get("content", "").strip()
                config = block.get("config")

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

                # --- image/chart ---
                if label in ["chart", "image", "seal"]:
                    image_name = block.get("content")
                    if not image_name:
                        return "% [Image not found]\n\n"

                    # get the absolute path
                    abs_image_path = abs_image_paths.get(image_name)
                    if not abs_image_path:
                        return f"% [Image path not found for {image_name}]\n\n"

                    para = doc.add_paragraph()
                    run = para.add_run()
                    run.add_picture(abs_image_path, width=Inches(5))
                    para.alignment = WD_ALIGN_PARAGRAPH.CENTER

                # --- table process ---
                elif label == "table" and content:
                    rows = (
                        parse_html_table(content)
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

                # --- other content ---
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
                    if label == "vision_footnote":
                        content = f"[footnote] {content}"
                    para = doc.add_paragraph(content)
                    set_paragraph_style(para, config)
            return doc

        fn = Path(self._get_input_fn())
        stem = fn.stem
        save_path = Path(save_path)
        save_file = save_path / f"{stem}.docx"
        # Determine whether the _to_word parameter originates from markdown.save_to_word or layout_parsing_result_.save_to_word
        sig = signature(getattr(self, "_to_word", None))
        params = sig.parameters.keys()
        if len(params) == 0:
            blocks = self._to_word()["word_blocks"]
            images = self._to_word()["images"]
            original_image_width = self._to_word().get("original_image_width", 500)
            abs_image_paths = save_images(images, Path(save_path))
            doc = blocks_to_word(blocks, original_image_width, abs_image_paths)
        else:
            doc = self._to_word(save_path.resolve())

        doc.save(save_file.as_posix())


class LatexMixin:
    def __init__(self, *args: list, **kwargs: dict):
        self._save_funcs.append(self.save_to_latex)

    @abstractmethod
    def _to_latex(self) -> Dict[str, Any]:
        """
        Convert the result to a LaTeX-compatible format.

        Returns:
            Dict[str, Any]: A dictionary containing LaTeX-compatible blocks and image data.
        """
        raise NotImplementedError

    @property
    def latex(self) -> Dict[str, Any]:
        """Property to access the LaTeX-compatible data.

        Returns:
            Dict[str, Any]: A dictionary containing LaTeX-compatible blocks and image data.
        """
        return self._to_latex()

    def save_to_latex(self, save_path, *args, **kwargs) -> None:
        from bs4 import BeautifulSoup

        """
        Save the LaTeX session dict to a .tex file (each page as separate file).
        """

        def save_images(image_list, base_save_path):
            """
            Save images to disk and return a mapping from original path to saved absolute path.
            """
            abs_image_paths: Dict[str, str] = {}
            image_dir = base_save_path / "imgs"
            image_dir.mkdir(parents=True, exist_ok=True)

            for item in image_list:
                img_path = item.get("path")
                img_obj = item.get("img")
                if not img_path or not img_obj:
                    continue

                img_name = Path(img_path).name
                save_path = image_dir / img_name
                img_obj.save(save_path)
                abs_image_paths[img_path] = str(save_path.resolve())
            return abs_image_paths

        def escape_latex(s: str) -> str:
            """
            Escape LaTeX special characters.
            """
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

        def escaped_paragraph_text(s: str) -> str:
            """
            Process regular paragraphs while preserving formulas.
            """
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

                temp = re.sub(
                    r"(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\\])", _hold, p, flags=re.DOTALL
                )
                temp = escape_latex(temp)
                for i, f in enumerate(placeholders):
                    temp = temp.replace(f"@@FORMULA{i}@@", f)
                processed.append("\\par " + temp)
            return "\n\n".join(processed) + "\n\n"

        def generate_image_latex(block, abs_image_paths) -> str:

            image_name = block.get("content")
            if not image_name:
                return "% [Image not found]\n\n"

            # get the absolute path
            abs_image_path = abs_image_paths.get(image_name)
            if not abs_image_path:
                return f"% [Image path not found for {image_name}]\n\n"

            return (
                f"\\begin{{figure}}[h]\n"
                f"\\centering\n"
                f"\\includegraphics[width=0.8\\linewidth]{{{abs_image_path}}}\n"
                f"\\end{{figure}}\n\n"
            )

        def generate_table_latex(block) -> str:
            content = block.get("content", "")
            if "<table" in content:
                soup = BeautifulSoup(content, "html.parser")
                rows = [
                    [
                        (
                            escape_latex(td.get_text(strip=True))
                            if not re.search(
                                r"(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])",
                                td.get_text(strip=True),
                            )
                            else td.get_text(strip=True)
                        )
                        for td in tr.find_all(["td", "th"])
                    ]
                    for tr in soup.find_all("tr")
                ]
            else:
                rows = [
                    [
                        (
                            escape_latex(c)
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

        def block_to_latex(block, abs_image_paths) -> str:
            label = block.get("type", "")
            content = block.get("content", "") or ""
            if label == "doc_title":
                return f"\\begin{{center}}\n{{\\Huge {escape_latex(content.strip())}}}\\end{{center}}\n\n"
            if label in ["header", "footer"]:
                return ""
            if label == "abstract":
                return f"\\begin{{abstract}}\n{escape_latex(content.strip())}\n\\end{{abstract}}\n\n"
            if label == "paragraph_title":
                return f"\\section*{{{escape_latex(content.strip())}}}\n\n"
            if label == "text":
                return escaped_paragraph_text(content)
            if label == "content":
                lines = [line.rstrip() for line in content.splitlines()]
                return (
                    "\n".join(
                        [escape_latex(line) + " \\\\" for line in lines if line.strip()]
                    )
                    + "\n\n"
                )
            if label == "formula":
                return f"\\[\n{content.strip()}\n\\]\n\n"
            if label == "algorithm":
                return "\\begin{verbatim}\n" + content + "\n\\end{verbatim}\n\n"
            if label in ["image", "chart", "seal"]:
                return generate_image_latex(block, abs_image_paths)
            if label == "table":
                return generate_table_latex(block)
            if label in ["figure_title", "table_title", "chart_title"]:
                return f"\\begin{{center}}\n{{\\small {escape_latex(content.strip())}}}\\end{{center}}\n\n"
            if label == "reference":
                lines = [line.strip() for line in content.split("\n") if line.strip()]
                bibitems = []
                for line in lines:
                    content = escape_latex(re.sub(r"^\[\d+\]\s*", "", line))
                    key = f"ref{abs(hash(line)) % 100000}"
                    bibitems.append(f"\\bibitem{{{key}}} {content}")
                return "\n".join(bibitems) + "\n", "\n".join(bibitems) + "\n"
            return f"% [Unknown block: {label}] {escape_latex(content)}\n\n"

        def blocks_to_latex(blocks, abs_image_paths) -> str:
            pages = {}
            for b in blocks:
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
                latex_lines.append(f"\\fancyhead[L]{{{escape_latex(page_header)}}}")
                latex_lines.append(f"\\fancyfoot[C]{{{escape_latex(page_footer)}}}\n")

                for block in page_blocks:
                    if block.get("type", "") == "reference_title" and not in_bib:
                        latex_lines.append("\\begin{thebibliography}{99}")
                        in_bib = True
                        continue
                    latex_lines.append(block_to_latex(block, abs_image_paths))

                latex_lines.append("\\clearpage\n")

            if in_bib:
                latex_lines.append("\\end{thebibliography}\n")

            latex_lines.append("\\end{document}")
            return "\n".join(latex_lines)

        fn = Path(self._get_input_fn())
        stem = fn.stem
        save_path = Path(save_path)
        save_file = save_path / f"{stem}.tex"

        sig = signature(getattr(self, "_to_latex", None))
        params = sig.parameters.keys()
        if len(params) == 0:
            blocks = self._to_latex()["latex_blocks"]
            images = self._to_latex()["images"]
            abs_image_paths = save_images(images, Path(save_path))
            latex = blocks_to_latex(blocks, abs_image_paths)
        else:
            latex = self._to_latex(save_path.resolve())

        os.makedirs(save_path, exist_ok=True)
        with open(save_file.as_posix(), "w", encoding="utf-8") as f:
            f.write(latex)


class StrMixin:
    """Mixin class for adding string conversion capabilities."""

    @property
    def str(self) -> Dict[str, str]:
        """Property to get the string representation of the result.

        Returns:
            Dict[str, str]: The string representation of the result.
        """

        return self._to_str()

    def _to_str(
        self,
    ):
        """Convert the given result data to a string representation.

        Args:
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.

        Returns:
            Dict[str, str]: The string representation of the result.
        """
        return {"res": self}

    def print(self) -> None:
        """Print the string representation of the result."""
        logging.info(self._to_str())


def _format_data(obj):
    """Helper function to format data into a JSON-serializable format.

    Args:
        obj: The object to be formatted.

    Returns:
        Any: The formatted object.
    """
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_format_data(item) for item in obj.tolist()]
    elif isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", force_ascii=False))
    elif isinstance(obj, Path):
        return obj.as_posix()
    elif isinstance(obj, dict):
        return dict({k: _format_data(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return [_format_data(i) for i in obj]
    else:
        return obj


class JsonMixin:
    """Mixin class for adding JSON serialization capabilities."""

    def __init__(self) -> None:
        self._json_writer = JsonWriter()
        self._save_funcs.append(self.save_to_json)

    def _to_json(self) -> Dict[str, Dict[str, Any]]:
        """Convert the object to a JSON-serializable format.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary representation of the object that is JSON-serializable.
        """

        return {"res": _format_data(copy.deepcopy(self))}

    @property
    def json(self) -> Dict[str, Dict[str, Any]]:
        """Property to get the JSON representation of the result.

        Returns:
            Dict[str, Dict[str, Any]]: The dict type JSON representation of the result.
        """

        return self._to_json()

    def save_to_json(
        self,
        save_path: str,
        indent: int = 4,
        ensure_ascii: bool = False,
        *args: List,
        **kwargs: Dict,
    ) -> None:
        """Save the JSON representation of the object to a file.

        Args:
            save_path (str): The path to save the JSON file. If the save path does not end with '.json', it appends the base name and suffix of the input path.
            indent (int): The number of spaces to indent for pretty printing. Default is 4.
            ensure_ascii (bool): If False, non-ASCII characters will be included in the output. Default is False.
            *args: Additional positional arguments to pass to the underlying writer.
            **kwargs: Additional keyword arguments to pass to the underlying writer.
        """

        def _is_json_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "application/json"

        json_data = self._to_json()
        if not _is_json_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in json_data:
                save_path = base_save_path / f"{stem}_{key}.json"
                self._json_writer.write(
                    save_path.as_posix(),
                    json_data[key],
                    indent=indent,
                    ensure_ascii=ensure_ascii,
                    *args,
                    **kwargs,
                )
        else:
            if len(json_data) > 1:
                logging.warning(
                    f"The result has multiple json files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._json_writer.write(
                save_path,
                json_data[list(json_data.keys())[0]],
                indent=indent,
                ensure_ascii=ensure_ascii,
                *args,
                **kwargs,
            )

    def _to_str(
        self,
        json_format: bool = False,
        indent: int = 4,
        ensure_ascii: bool = False,
    ):
        """Convert the given result data to a string representation.
        Args:
            data (dict): The data would be converted to str.
            json_format (bool): If True, return a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        Returns:
            Dict[str, str]: The string representation of the result.
        """
        if json_format:
            return json.dumps(
                _format_data({"res": self}), indent=indent, ensure_ascii=ensure_ascii
            )
        else:
            return {"res": self}

    def print(
        self, json_format: bool = False, indent: int = 4, ensure_ascii: bool = False
    ) -> None:
        """Print the string representation of the result.

        Args:
            json_format (bool): If True, print a JSON formatted string. Default is False.
            indent (int): Number of spaces to indent for JSON formatting. Default is 4.
            ensure_ascii (bool): If True, ensure all characters are ASCII. Default is False.
        """
        str_ = self._to_str(
            json_format=json_format, indent=indent, ensure_ascii=ensure_ascii
        )
        logging.info(str_)


class Base64Mixin:
    """Mixin class for adding Base64 encoding capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes the Base64Mixin.

        Args:
            *args: Positional arguments to pass to the TextWriter.
            **kwargs: Keyword arguments to pass to the TextWriter.
        """
        self._base64_writer = TextWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_base64)

    @abstractmethod
    def _to_base64(self) -> Dict[str, str]:
        """Abstract method to convert the result to Base64.

        Returns:
            Dict[str, str]: The str type Base64 representation result.
        """
        raise NotImplementedError

    @property
    def base64(self) -> Dict[str, str]:
        """
        Property that returns the Base64 encoded content.

        Returns:
            Dict[str, str]: The base64 representation of the result.
        """
        return self._to_base64()

    def save_to_base64(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the Base64 encoded content to the specified path.

        Args:
            save_path (str): The path to save the base64 representation result. If the save path does not end with '.b64', it appends the base name and suffix of the input path.

            *args: Additional positional arguments that will be passed to the base64 writer.
            **kwargs: Additional keyword arguments that will be passed to the base64 writer.
        """
        base64 = self._to_base64()
        if not str(save_path).lower().endswith((".b64")):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in base64:
                save_path = base_save_path / f"{stem}_{key}.b64"
                self._base64_writer.write(
                    save_path.as_posix(), base64[key], *args, **kwargs
                )
        else:
            if len(base64) > 1:
                logging.warning(
                    f"The result has multiple base64 files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._base64_writer.write(
                save_path, base64[list(base64.keys())[0]], *args, **kwargs
            )


class ImgMixin:
    """Mixin class for adding image handling capabilities."""

    def __init__(self, backend: str = "pillow", *args: List, **kwargs: Dict) -> None:
        """Initializes ImgMixin.

        Args:
            backend (str): The backend to use for image processing. Defaults to "pillow".
            *args: Additional positional arguments to pass to the ImageWriter.
            **kwargs: Additional keyword arguments to pass to the ImageWriter.
        """
        self._img_writer = ImageWriter(backend=backend, *args, **kwargs)
        self._save_funcs.append(self.save_to_img)

    @abstractmethod
    def _to_img(self) -> Dict[str, Image.Image]:
        """Abstract method to convert the result to an image.

        Returns:
            Dict[str, Image.Image]: The image representation result.
        """
        raise NotImplementedError

    @property
    def img(self) -> Dict[str, Image.Image]:
        """Property to get the image representation of the result.

        Returns:
            Dict[str, Image.Image]: The image representation of the result.
        """
        return self._to_img()

    def save_to_img(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the image representation of the result to the specified path.

        Args:
            save_path (str): The path to save the image. If the save path does not end with .jpg or .png, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the image writer.
            **kwargs: Additional keyword arguments that will be passed to the image writer.
        """

        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("image/")

        img = self._to_img()
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in img:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                self._img_writer.write(save_path.as_posix(), img[key], *args, **kwargs)
        else:
            if len(img) > 1:
                logging.warning(
                    f"The result has multiple img files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._img_writer.write(save_path, img[list(img.keys())[0]], *args, **kwargs)


class CSVMixin:
    """Mixin class for adding CSV handling capabilities."""

    def __init__(self, backend: str = "pandas", *args: List, **kwargs: Dict) -> None:
        """Initializes the CSVMixin.

        Args:
            backend (str): The backend to use for CSV operations (default is "pandas").
            *args: Optional positional arguments to pass to the CSVWriter.
            **kwargs: Optional keyword arguments to pass to the CSVWriter.
        """
        self._csv_writer = CSVWriter(backend=backend, *args, **kwargs)
        if not hasattr(self, "_save_funcs"):
            self._save_funcs = []
        self._save_funcs.append(self.save_to_csv)

    @property
    def csv(self) -> Dict[str, pd.DataFrame]:
        """Property to get the pandas Dataframe representation of the result.

        Returns:
            Dict[str, pd.DataFrame]: The pandas.DataFrame representation of the result.
        """
        return self._to_csv()

    @abstractmethod
    def _to_csv(self) -> Dict[str, pd.DataFrame]:
        """Abstract method to convert the result to pandas.DataFrame.

        Returns:
            Dict[str, pd.DataFrame]: The pandas.DataFrame representation result.
        """
        raise NotImplementedError

    def save_to_csv(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the result to a CSV file.

        Args:
            save_path (str): The path to save the CSV file. If the path does not end with ".csv",
                the stem of the input path attribute (self['input_path']) will be used as the filename.
            *args: Optional positional arguments to pass to the CSV writer's write method.
            **kwargs: Optional keyword arguments to pass to the CSV writer's write method.
        """

        def _is_csv_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "text/csv"

        csv = self._to_csv()
        if not _is_csv_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in csv:
                save_path = base_save_path / f"{stem}_{key}.csv"
                self._csv_writer.write(save_path.as_posix(), csv[key], *args, **kwargs)
        else:
            if len(csv) > 1:
                logging.warning(
                    f"The result has multiple csv files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._csv_writer.write(save_path, csv[list(csv.keys())[0]], *args, **kwargs)


class HtmlMixin:
    """Mixin class for adding HTML handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """
        Initializes the HTML writer and appends the save_to_html method to the save functions list.

        Args:
            *args: Positional arguments passed to the HtmlWriter.
            **kwargs: Keyword arguments passed to the HtmlWriter.
        """
        self._html_writer = HtmlWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_html)

    @property
    def html(self) -> Dict[str, str]:
        """Property to get the HTML representation of the result.

        Returns:
            str: The str type HTML representation of the result.
        """
        return self._to_html()

    @abstractmethod
    def _to_html(self) -> Dict[str, str]:
        """Abstract method to convert the result to str type HTML representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_html(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation of the object to the specified path.

        Args:
            save_path (str): The path to save the HTML file.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        def _is_html_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "text/html"

        html = self._to_html()
        if not _is_html_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in html:
                save_path = base_save_path / f"{stem}_{key}.html"
                self._html_writer.write(
                    save_path.as_posix(), html[key], *args, **kwargs
                )
        else:
            if len(html) > 1:
                logging.warning(
                    f"The result has multiple html files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._html_writer.write(
                save_path, html[list(html.keys())[0]], *args, **kwargs
            )


class XlsxMixin:
    """Mixin class for adding XLSX handling capabilities."""

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        """Initializes the XLSX writer and appends the save_to_xlsx method to the save functions.

        Args:
            *args: Positional arguments to be passed to the XlsxWriter constructor.
            **kwargs: Keyword arguments to be passed to the XlsxWriter constructor.
        """
        self._xlsx_writer = XlsxWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_xlsx)

    @property
    def xlsx(self) -> Dict[str, str]:
        """Property to get the XLSX representation of the result.

        Returns:
            Dict[str, str]: The str type XLSX representation of the result.
        """
        return self._to_xlsx()

    @abstractmethod
    def _to_xlsx(self) -> Dict[str, str]:
        """Abstract method to convert the result to str type XLSX representation.

        Returns:
            Dict[str, str]: The str type HTML representation result.
        """
        raise NotImplementedError

    def save_to_xlsx(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the HTML representation to an XLSX file.

        Args:
            save_path (str): The path to save the XLSX file. If the path does not end with ".xlsx",
                             the filename will be set to the stem of the input path with ".xlsx" extension.
            *args: Additional positional arguments to pass to the XLSX writer.
            **kwargs: Additional keyword arguments to pass to the XLSX writer.
        """

        def _is_xlsx_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return (
                mime_type is not None
                and mime_type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        xlsx = self._to_xlsx()
        if not _is_xlsx_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in xlsx:
                save_path = base_save_path / f"{stem}_{key}.xlsx"
                self._xlsx_writer.write(
                    save_path.as_posix(), xlsx[key], *args, **kwargs
                )
        else:
            if len(xlsx) > 1:
                logging.warning(
                    f"The result has multiple xlsx files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            self._xlsx_writer.write(
                save_path, xlsx[list(xlsx.keys())[0]], *args, **kwargs
            )


class VideoMixin:
    """Mixin class for adding Video handling capabilities."""

    def __init__(self, backend: str = "opencv", *args: List, **kwargs: Dict) -> None:
        """Initializes VideoMixin.

        Args:
            backend (str): The backend to use for video processing. Defaults to "opencv".
            *args: Additional positional arguments to pass to the VideoWriter.
            **kwargs: Additional keyword arguments to pass to the VideoWriter.
        """
        self._backend = backend
        self._save_funcs.append(self.save_to_video)

    @abstractmethod
    def _to_video(self) -> Dict[str, np.array]:
        """Abstract method to convert the result to a video.

        Returns:
            Dict[str, np.array]: The video representation result.
        """
        raise NotImplementedError

    @property
    def video(self) -> Dict[str, np.array]:
        """Property to get the video representation of the result.

        Returns:
            Dict[str, np.array]: The video representation of the result.
        """
        return self._to_video()

    def save_to_video(self, save_path: str, *args: List, **kwargs: Dict) -> None:
        """Saves the video representation of the result to the specified path.

        Args:
            save_path (str): The path to save the video. If the save path does not end with .mp4 or .avi, it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the video writer.
            **kwargs: Additional keyword arguments that will be passed to the video writer.
        """

        def _is_video_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("video/")

        video_writer = VideoWriter(backend=self._backend, *args, **kwargs)
        video = self._to_video()
        if not _is_video_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            suffix = fn.suffix if _is_video_file(fn) else ".mp4"
            base_save_path = Path(save_path)
            for key in video:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                video_writer.write(save_path.as_posix(), video[key], *args, **kwargs)
        else:
            if len(video) > 1:
                logging.warning(
                    f"The result has multiple video files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            video_writer.write(save_path, video[list(video.keys())[0]], *args, **kwargs)


class MarkdownMixin:
    """Mixin class for adding Markdown handling capabilities."""

    MARKDOWN_SAVE_KEYS = ["markdown_texts"]

    def __init__(self, *args: list, **kwargs: dict):
        """Initializes the Markdown writer and appends the save_to_markdown method to the save functions.

        Args:
            *args: Positional arguments to be passed to the MarkdownWriter constructor.
            **kwargs: Keyword arguments to be passed to the MarkdownWriter constructor.
        """
        self._markdown_writer = MarkdownWriter(*args, **kwargs)
        self._img_writer = ImageWriter(*args, **kwargs)
        self._save_funcs.append(self.save_to_markdown)

    @abstractmethod
    def _to_markdown(
        self, pretty=True, show_formula_number=False
    ) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Convert the result to markdown format.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.

        Returns:
            Dict[str, Union[str, Dict[str, Any]]]: A dictionary containing markdown text and image data.
        """
        raise NotImplementedError

    @property
    def markdown(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Property to access the markdown data.

        Returns:
            Dict[str, Union[str, Dict[str, Any]]]: A dictionary containing markdown text and image data.
        """
        return self._to_markdown()

    def save_to_markdown(
        self, save_path, pretty=True, show_formula_number=False, *args, **kwargs
    ) -> None:
        """Save the markdown data to a file.

        Args:
            save_path (Union[str, Path]): The path where the markdown file will be saved.
            *args: Additional positional arguments for saving.
            **kwargs: Additional keyword arguments for saving.
        """

        def _is_markdown_file(file_path) -> bool:
            """Check if a file is a markdown file based on its extension or MIME type.

            Args:
                file_path (Union[str, Path]): The path to the file.

            Returns:
                bool: True if the file is a markdown file, False otherwise.
            """
            markdown_extensions = {".md", ".markdown", ".mdown", ".mkd"}
            _, ext = os.path.splitext(str(file_path))
            if ext.lower() in markdown_extensions:
                return True
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type == "text/markdown"

        if not _is_markdown_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_markdown_file(fn) else ".md"
            stem = fn.stem
            base_save_path = Path(save_path)
            save_path = base_save_path / f"{stem}{suffix}"
            self.save_path = save_path
        else:
            self.save_path = save_path
        self._save_data(
            self._markdown_writer.write,
            self._img_writer.write,
            self.save_path,
            self._to_markdown(pretty=pretty, show_formula_number=show_formula_number),
            *args,
            **kwargs,
        )

    def _save_data(
        self,
        save_mkd_func: Callable,
        save_img_func: Callable,
        save_path: Union[str, Path],
        data: Optional[Dict[str, Union[str, Dict[str, Any]]]],
        *args,
        **kwargs,
    ) -> None:
        """Internal method to save markdown and image data.

        Args:
            save_mkd_func (Callable): Function to save markdown text.
            save_img_func (Callable): Function to save image data.
            save_path (Union[str, Path]): The base path where the data will be saved.
            data (Optional[Dict[str, Union[str, Dict[str, Any]]]]): The markdown data to save.
            *args: Additional positional arguments for saving.
            **kwargs: Additional keyword arguments for saving.
        """
        save_path = Path(save_path)
        if data is None:
            return
        for key, value in data.items():
            if key in self.MARKDOWN_SAVE_KEYS:
                save_mkd_func(save_path.as_posix(), value, *args, **kwargs)
            if isinstance(value, dict):
                base_save_path = save_path.parent
                for img_path, img_data in value.items():
                    save_img_func(
                        (base_save_path / img_path).as_posix(),
                        img_data,
                        *args,
                        **kwargs,
                    )
